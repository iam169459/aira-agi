"""Text-to-Speech engine, from scratch.

Pipeline: normalize text -> grapheme-to-phoneme -> prosody model (stress,
durations, F0 contour, pausing) -> formant synthesizer -> WAV.

Dependencies: numpy + standard library only.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
import wave

import numpy as np

from . import g2p as g2p_mod
from . import phonemes as ph
from .normalize import normalize_text
from .synth import FormantSynthesizer


def save_wav(path, samples, sample_rate):
    """Write float samples ([-1, 1]) as a 16-bit mono WAV file."""
    samples = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def load_wav(path):
    """Read a WAV file into float samples [-1, 1] and its sample rate.

    Accepts mono/stereo, 8/16/24/32-bit PCM and 32-bit float.
    """
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        width = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) |
                (b[:, 1].astype(np.int32) << 8) |
                (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        data = ints.astype(np.float64) / float(1 << 23)
    elif width == 4:
        vals = np.frombuffer(raw, dtype=np.int32)
        if np.any(np.abs(vals) > (1 << 30)):
            data = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        else:
            data = vals.astype(np.float64) / float(1 << 31)
        if np.max(np.abs(data)) < 1.5 and np.all(np.abs(data) <= 1.0) and \
                np.max(np.abs(data)) > 0:
            pass
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr


def resample(samples, sr_from, sr_to):
    if sr_from == sr_to or len(samples) == 0:
        return np.asarray(samples, dtype=np.float64)
    duration = len(samples) / sr_from
    n_out = max(1, int(duration * sr_to))
    x_old = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples)


BASE_DURATION = {
    "vowel": 0.115, "diphthong": 0.165, "nasal": 0.075, "liquid": 0.070,
    "glide": 0.065, "fricative": 0.095, "stop": 0.075, "affricate": 0.14,
    "silent": 0.06,
}

PAUSES = {"SIL": 0.30, "COMMA": 0.17, "WBRK": 0.062, "BRK": 0.09}


class TextToSpeech:
    """From-scratch synthesizer with a young-female-style default voice."""

    def __init__(self, sample_rate=22050, f0=185.0, f0_range=0.20,
                 rate=1.0, breathiness=0.018, voice=None):
        self.sample_rate = sample_rate
        self.rate = max(0.5, min(2.0, rate))
        self.synth = FormantSynthesizer(sample_rate=sample_rate, base_f0=f0,
                                        f0_range=f0_range,
                                        breathiness=breathiness)
        self.speed_factor = 1.0 + (150.0 - 165.0) / 165.0
        self.voice_names = {
            "female": {"f0": 190.0},
            "male": {"f0": 115.0},
            "child": {"f0": 250.0},
            "neutral": {"f0": 150.0},
        }
        if voice and voice in self.voice_names:
            self.synth.base_f0 = self.voice_names[voice]["f0"]

    # ------------------------------------------------------------------
    def phonemes(self, text):
        text = normalize_text(text)
        return g2p_mod.g2p(text)

    def plan(self, text, emotion="neutral"):
        """Build synthesis events: list of dicts
        {ph, start, dur, f0_start, f0_end, amp, word_index}"""
        phonemes, n_words = self.phonemes(text)
        if not phonemes:
            return []
        events = self._plan_events(phonemes, text, emotion)
        return events

    def _plan_events(self, phonemes, text="", emotion="neutral"):
        base_f0 = self.synth.base_f0
        f0_span = self.synth.f0_range
        speed = self.rate

        stress_flags = self._mark_stress(phonemes)

        events = []
        t = 0.0
        vowel_positions = [i for i, p in enumerate(phonemes)
                           if ph.phone_group(p) in ("vowel", "diphthong")]
        n_voiced = max(1, len(vowel_positions))
        voiced_seen = 0

        end_pitch_drop = 1.0 - 0.12 * f0_span * 5
        for idx, p in enumerate(phonemes):
            if p in PAUSES:
                dur = PAUSES[p] / speed
                if p == "WBRK":
                    next_ph = phonemes[idx + 1] if idx + 1 < len(phonemes) \
                        else "SIL"
                    if ph.phone_group(next_ph) in ("stop", "affricate"):
                        events.append({"ph": "SP", "start": t, "dur": dur,
                                       "f0_start": base_f0,
                                       "f0_end": base_f0, "amp": 0.0})
                    t += dur
                else:
                    events.append({"ph": p, "start": t, "dur": dur,
                                   "f0_start": base_f0, "f0_end": base_f0,
                                   "amp": 0.0})
                    t += dur
                continue
            group = ph.phone_group(p)
            if group in ("stop",):
                dur = 0.030 + 0.045 / speed
                if p in ("B", "D", "G"):
                    dur = 0.055 / speed
            elif group == "affricate":
                dur = 0.13 / speed
            else:
                dur = BASE_DURATION[group] / speed
                stressed = stress_flags[idx]
                if stressed:
                    dur *= 1.25
                elif group in ("vowel", "diphthong"):
                    dur *= 0.82
            amp = 0.95 if stress_flags[idx] else 0.78

            if p in ph.VOWEL_GROUP:
                voiced_seen += 1
                decl = 1.0 - 0.14 * (voiced_seen / n_voiced)
                accent = 1.05 if stress_flags[idx] else 0.98
                f0 = base_f0 * decl * accent
            else:
                f0 = base_f0 * 0.99
            f0_start = f0
            f0_end = f0 * (0.995 if group in ("vowel", "diphthong") else 1.0)

            events.append({"ph": p, "start": t, "dur": dur,
                           "f0_start": f0_start, "f0_end": f0_end,
                           "amp": amp})
            t += dur

        contour = self._sentence_contour(text, events, base_f0, f0_span)
        for e, (f0s, f0e) in zip(events, contour):
            e["f0_start"] *= f0s
            e["f0_end"] *= f0e
        return events

    def _sentence_contour(self, text, events, base_f0, span):
        """Global intonation: question rise, statement fall, exclamation."""
        text = text.strip()
        is_question = text.endswith("?")
        is_exclaim = text.endswith("!")
        n = len(events)
        factors = []
        for i in range(n):
            start_s = 1.0
            end_s = 1.0
            if is_question:
                frac = i / max(n - 1, 1)
                start_s = 1.0 + 0.28 * span * frac * 4
                end_s = 1.0 + 0.30 * span * frac * 4
            elif is_exclaim:
                frac = i / max(n - 1, 1)
                bump = 1.0 + 0.12 * np.sin(frac * np.pi) * span * 4
                start_s = bump
                end_s = bump * 0.99
            else:
                frac = i / max(n - 1, 1)
                start_s = 1.0 - 0.10 * frac
                end_s = 1.0 - 0.11 * frac
            factors.append((start_s, end_s))
        if events:
            if is_question:
                factors[-1] = (factors[-1][0], 1.0 + 0.34 * span * 4)
            else:
                factors[-1] = (factors[-1][0], 0.86)
        return factors

    def _mark_stress(self, phonemes):
        """First vowel of each word group receives primary stress."""
        stress = [False] * len(phonemes)
        in_word = False
        first_vowel_done = False
        for i, p in enumerate(phonemes):
            if p in PAUSES:
                in_word = False
                first_vowel_done = False
                continue
            group = ph.phone_group(p)
            if not in_word:
                if group in ("vowel", "diphthong") and not first_vowel_done:
                    stress[i] = True
                    first_vowel_done = True
                    in_word = True
                elif group in ("vowel", "diphthong"):
                    in_word = True
            else:
                if group in ("vowel", "diphthong"):
                    first_vowel_done = True
        return stress

    # ------------------------------------------------------------------
    def synthesize(self, text, emotion="neutral"):
        events = self.plan(text, emotion)
        if not events:
            return np.zeros(0, dtype=np.float32)
        return self.synth.synthesize(events)

    def synthesize_to_file(self, text, path, emotion="neutral"):
        audio = self.synthesize(text, emotion)
        save_wav(path, audio, self.sample_rate)
        return path, len(audio) / self.sample_rate

    def speak_to_tempfile(self, text, emotion="neutral"):
        fd, path = tempfile.mkstemp(prefix="aira_tts_", suffix=".wav")
        os.close(fd)
        self.synthesize_to_file(text, path, emotion)
        return path

    def speak(self, text, emotion="neutral", block=True):
        """Render and play through the OS audio player (best effort)."""
        path = self.speak_to_tempfile(text, emotion)
        try:
            if block:
                self.play(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    def play(path):
        if sys.platform == "darwin":
            cmd = ["afplay", path]
        elif sys.platform.startswith("linux"):
            for player in ("paplay", "aplay", "play"):
                if subprocess.call(["which", player],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL) == 0:
                    cmd = [player, path]
                    break
            else:
                raise RuntimeError("No audio player found (paplay/aplay/play)")
        elif sys.platform == "win32":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)

"""Source-filter formant synthesizer, from scratch.

Voiced sounds: a shaped glottal flow-derivative waveform (Rosenberg-style
pulse with cycle-to-cycle jitter and shimmer) plus aspiration noise keyed to
the glottal open phase drives a bank of damped-sinusoid resonators. Each
time-varying formant segment filters its slice of the excitation by
convolving with the resonators' analytic impulse responses and overlap-adding
the decay tails, so formant targets move smoothly between phones without any
per-sample Python recursion.

Unvoiced sounds: shaped white noise (FFT band-pass) driven by a fricative /
burst / aspiration spec.
"""
import numpy as np

from . import phonemes as ph

FORMANT_GAINS = (1.0, 0.52, 0.24)


class FormantSynthesizer:
    def __init__(self, sample_rate=22050, base_f0=185.0, f0_range=0.18,
                 breathiness=0.018, open_quotient=0.60, tilt=1200.0):
        self.sr = sample_rate
        self.base_f0 = base_f0
        self.f0_range = f0_range
        self.breathiness = breathiness
        self.open_quotient = open_quotient
        self.tilt = tilt
        self.transition_ms = 40.0
        self.ramp_ms = 7.0
        self._kernel_cache = {}

    # ------------------------------------------------------------------
    def synthesize(self, events):
        """events: list of dicts {ph, start, dur, f0_start, f0_end, amp}
        (times in seconds). Returns float32 waveform in [-1, 1]."""
        if not events:
            return np.zeros(0, dtype=np.float32)
        total = max(e["start"] + e["dur"] for e in events) + 0.05
        n_total = int(total * self.sr)
        audio = np.zeros(n_total, dtype=np.float64)

        src, open_env = self._glottal_source(events, n_total)
        env = self._event_envelope(events, n_total)
        nseed = np.random.default_rng(20240517)
        breath = (nseed.uniform(-1, 1, n_total) * open_env *
                  self.breathiness * 3.0)

        segments = self._formant_schedule(events)
        audio = self._render_all(segments, src, breath, env, n_total)
        audio = self._finish(audio)
        return audio.astype(np.float32)

    def _render_all(self, segments, src, breath, env, n_total):
        audio = np.zeros(n_total)
        for seg in segments:
            s0 = int(seg["start"] * self.sr)
            s1 = min(s0 + max(1, int(seg["dur"] * self.sr)), n_total)
            if s1 <= s0:
                continue
            if seg["voiced"] and seg["formants"] is not None:
                self._render_voiced(seg, s0, s1, src, breath, env, audio)
            if seg["noise"] is not None:
                audio[s0:s1] += self._render_noise(seg)
        return audio

    # ------------------------------------------------------------------
    def _glottal_source(self, events, n_total):
        """Shaped glottal flow derivative with jitter/shimmer, plus a per-
        sample envelope of glottal openness used to gate aspiration noise."""
        src = np.zeros(n_total)
        open_env = np.full(n_total, 0.10)
        for e in events:
            if not self._event_voiced(e):
                continue
            s0 = int(e["start"] * self.sr)
            n = max(1, int(e["dur"] * self.sr))
            f0a, f0b, amp = e["f0_start"], e["f0_end"], e["amp"]
            rng = np.random.default_rng(int(s0) + 7919)
            tloc = 0.0
            guard = 0
            while tloc < n and guard < 200000:
                guard += 1
                frac = min(tloc / max(n, 1), 1.0)
                f0 = f0a + (f0b - f0a) * frac
                jitter = 1.0 + rng.uniform(-0.022, 0.022)
                T0 = self.sr / max(f0, 40.0) * jitter
                pos = s0 + int(round(tloc))
                if pos >= n_total:
                    break
                olen = min(max(2, int(T0 * self.open_quotient)),
                           n_total - pos)
                tau = np.arange(olen) / max(olen - 1, 1)
                shape = (6.0 * tau - 6.0 * tau * tau) / 1.5
                shimmer = amp * (0.93 + 0.14 * rng.uniform())
                src[pos:pos + olen] += shape * shimmer
                open_env[pos:pos + olen] = 1.0
                tloc += T0
        return src, open_env

    @staticmethod
    def _event_voiced(e):
        return e["ph"] in ph.VOICED

    def _event_envelope(self, events, n_total):
        """Per-event half-cosine amplitude gesture applied at the glottis."""
        env = np.zeros(n_total)
        r = max(1, int(self.ramp_ms / 1000.0 * self.sr))
        for e in events:
            if e["amp"] <= 0.0:
                continue
            s0 = int(e["start"] * self.sr)
            n = max(1, int(e["dur"] * self.sr))
            s1 = min(s0 + n, n_total)
            if s1 <= s0:
                continue
            ramp = min(r, (s1 - s0) // 2)
            gesture = np.ones(s1 - s0)
            if ramp > 0:
                edge = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
                gesture[:ramp] = edge
                gesture[-ramp:] = edge[::-1]
            np.maximum(env[s0:s1], gesture, out=env[s0:s1])
        return env

    # ------------------------------------------------------------------
    def _formant_schedule(self, events):
        """Build formant control segments from events, with smooth multi-step
        glide ramps between consecutive voiced targets."""
        segments = []
        prev_formants = None
        for e in events:
            for s in self._event_segments(e, prev_formants):
                segments.append(s)
                if s["formants"] is not None:
                    prev_formants = s["formants"]
        return segments

    def _event_segments(self, e, prev_formants):
        name = e["ph"]
        group = ph.phone_group(name)
        start = e["start"]
        dur = e["dur"]
        segs = []
        noise = None
        formants = None

        if group == "vowel":
            spec = ph.VOWELS[name]
            formants = {"f": list(spec["f"]), "bw": list(spec["bw"])}
        elif group == "diphthong":
            spec = ph.DIPHTHONGS[name]
            a, b = spec["from"], spec["to"]
            p1 = [a[k] + (b[k] - a[k]) * 0.45 for k in range(3)]
            p2 = [a[k] + (b[k] - a[k]) * 0.78 for k in range(3)]
            return self._voiced_piecewise(e, prev_formants,
                                          [(0.35, p1), (0.70, p2), (1.0, b)])
        elif group in ("nasal", "liquid", "glide"):
            table = {"nasal": ph.NASALS, "liquid": ph.LIQUIDS,
                     "glide": ph.GLIDES}[group]
            spec = table[name]
            formants = {"f": list(spec["f"]), "bw": list(spec["bw"])}
        elif group == "fricative":
            spec = ph.FRICATIVES[name]
            noise = {"kind": "fricative", "spec": spec}
            if spec.get("voiced"):
                formants = {"f": [250, 1200, 2400], "bw": [150, 300, 400]}
            else:
                segs.append(self._simple(start, dur, None, noise, e))
                return segs
        elif group == "stop":
            spec = ph.STOPS[name]
            noise = {"kind": "stop", "spec": spec}
            if spec.get("voiced"):
                formants = {"f": [250, 1100, 2300], "bw": [150, 350, 450]}
            else:
                segs.append(self._simple(start, dur, None, noise, e))
                return segs
        elif group == "affricate":
            spec = ph.AFFRICATES[name]
            stop_spec = ph.STOPS[spec["stop"]]
            fric_spec = ph.FRICATIVES[spec["fric"]]
            closure = dur * 0.45
            segs.append(self._simple(start, closure, None,
                                     {"kind": "closure", "spec": stop_spec},
                                     e))
            segs.append(self._simple(start + closure, dur - closure, None,
                                     {"kind": "fricative",
                                      "spec": fric_spec}, e))
            return segs
        else:
            segs.append(self._simple(start, dur, None,
                                     {"kind": "silence",
                                      "spec": {"level": 0}}, e))
            return segs

        return self._voiced_piecewise_with_noise(e, prev_formants, formants,
                                                 noise)

    def _voiced_piecewise(self, e, prev_formants, points):
        """points: [(frac, formants), ...]: fraction of duration at which the
        formants reach that value. Emits a multi-step glide plus steady
        pieces."""
        segs = []
        dur = e["dur"]
        start = e["start"]
        cursor = 0.0
        fracs = [p[0] for p in points]
        vals = [p[1] for p in points]

        lead = self.transition_ms / 1000.0
        if prev_formants is not None and lead < dur * 0.5:
            segs.extend(self._glide_steps(start, lead, prev_formants["f"],
                                          vals[0], e))
            cursor = lead
        for idx in range(len(points)):
            end = fracs[idx] * dur
            if end <= cursor:
                continue
            segs.append(self._make_seg(start, cursor, end - cursor, vals[idx],
                                       self._default_bw(), e))
            cursor = end
        if cursor < dur:
            segs.append(self._make_seg(start, cursor, dur - cursor,
                                       vals[-1], self._default_bw(), e))
        return segs

    def _voiced_piecewise_with_noise(self, e, prev_formants, formants, noise):
        segs = []
        start, dur = e["start"], e["dur"]
        cursor = 0.0
        lead = self.transition_ms / 1000.0
        if prev_formants is not None and lead < dur * 0.6:
            segs.extend(self._glide_steps(start, lead, prev_formants["f"],
                                          formants["f"], e))
            cursor = lead
        segs.append(self._make_seg(start, cursor, dur - cursor,
                                   formants["f"], formants["bw"], e))
        for s in segs:
            s["noise"] = noise
        return segs

    def _glide_steps(self, start, lead, from_f, to_f, event, steps=3):
        """Interpolated formant ramp broken into short pieces so formant
        motion sounds continuous instead of steppy."""
        segs = []
        step_dur = lead / steps
        for i in range(steps):
            mid = (i + 0.5) / steps
            f = [from_f[k] + (to_f[k] - from_f[k]) * mid for k in range(3)]
            segs.append(self._make_seg(start, i * step_dur, step_dur, f,
                                       self._default_bw(), event))
        return segs

    def _default_bw(self):
        return [80, 110, 160]

    def _make_seg(self, start, off, dur, formants, bw, event):
        return {
            "start": start + off,
            "dur": max(dur, 1.0 / self.sr),
            "formants": {"f": list(formants), "bw": list(bw)},
            "voiced": True,
            "noise": None,
            "event": event,
            "amp": event["amp"],
        }

    def _simple(self, start, dur, formants, noise, event):
        return {
            "start": start,
            "dur": max(dur, 1.0 / self.sr),
            "formants": formants,
            "voiced": formants is not None,
            "noise": noise,
            "event": event,
            "amp": event["amp"],
        }

    # ------------------------------------------------------------------
    def _render_voiced(self, seg, s0, s1, src, breath, env, audio):
        exc = (src[s0:s1] + breath[s0:s1]) * env[s0:s1] * seg["amp"]
        if not len(exc):
            return
        f = seg["formants"]["f"]
        bw = seg["formants"]["bw"]
        convs = [np.convolve(exc, self._kernel(f[k], bw[k]))
                 for k in range(3)]
        out = np.zeros(max(len(c) for c in convs))
        for k, c in enumerate(convs):
            out[:len(c)] += c * FORMANT_GAINS[k]
        end = min(s0 + len(out), len(audio))
        audio[s0:end] += out[:end - s0]

    def _kernel(self, f, bw):
        key = (round(f, 1), round(bw, 1))
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached
        t_end = np.log(500.0) / (np.pi * max(bw, 30.0))
        n = max(int(t_end * self.sr), 2)
        t = np.arange(n) / self.sr
        kern = np.exp(-np.pi * bw * t) * np.sin(2 * np.pi * f * t)
        self._kernel_cache[key] = kern
        return kern

    def _render_noise(self, seg):
        spec = seg["noise"]["spec"]
        kind = seg["noise"]["kind"]
        s0 = int(seg["start"] * self.sr)
        n_samples = max(1, int(seg["dur"] * self.sr))
        if kind in ("silence", "closure"):
            return np.zeros(n_samples)
        rng = np.random.default_rng(s0 * 31 + 7)
        noise = rng.uniform(-1, 1, n_samples)
        env = self._amp_envelope(n_samples)
        if kind == "fricative" and "next_vowel" in spec:
            return (self._fft_bandpass(noise, 1400, 2000) * 0.20 * env +
                    self._fft_bandpass(noise, 500, 900) * 0.10 * env)
        if kind == "fricative":
            center, width = spec["noise"]
            sig = self._fft_bandpass(noise, center, width)
            sig += self._fft_bandpass(noise, center * 1.8, width * 1.6) * 0.5
            return sig * spec.get("level", 0.2) * env
        if kind == "stop":
            out = np.zeros(n_samples)
            burst_n = min(max(1, int(0.010 * self.sr)), n_samples)
            sig = self._fft_bandpass(noise[:burst_n], spec["burst"],
                                     spec["bw"]) * 1.4
            out[:burst_n] = sig
            asp_ms = spec.get("aspirate", 0)
            if asp_ms > 0:
                asp_n = min(int(asp_ms / 1000.0 * self.sr),
                            n_samples - burst_n)
                if asp_n > 0:
                    asp = self._fft_bandpass(
                        noise[burst_n:burst_n + asp_n], 1200, 1800) * 0.20
                    out[burst_n:burst_n + asp_n] += \
                        asp * env[burst_n:burst_n + asp_n]
            return out * env
        return noise * 0.1 * env

    def _fft_bandpass(self, noise, center, width):
        n = len(noise)
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sr)
        mag = self._resonator_mag(freqs, center, width)
        mag = mag / (mag.max() + 1e-9)
        return np.fft.irfft(spec * mag, n)

    @staticmethod
    def _resonator_mag(freqs, center, width):
        q = max(center / max(width, 50.0), 0.3)
        x = freqs / max(center, 1.0)
        return 1.0 / np.sqrt((1 - x * x) ** 2 + (x / q) ** 2)

    def _amp_envelope(self, n_samples):
        ramp = min(int(self.ramp_ms / 1000.0 * self.sr), n_samples // 2)
        env = np.ones(n_samples)
        if ramp > 0:
            edge = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
            env[:ramp] = edge
            env[-ramp:] = edge[::-1]
        return env

    # ------------------------------------------------------------------
    def _finish(self, audio):
        audio = audio - np.mean(audio)
        if self.tilt > 0 and len(audio) > 64:
            spec = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sr)
            tilt = 1.0 / np.sqrt(1.0 + (freqs / self.tilt) ** 2)
            audio = np.fft.irfft(spec * tilt, len(audio))
        peak = np.max(np.abs(audio)) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak * 0.92
        return np.clip(audio, -1.0, 1.0)

"""Two-way interruption for Aira's voice conversations.

Direction 1 — the user interrupts Aira (barge-in):
    While she is speaking, her microphone is watched. If the user's
    voice rises above the noise floor for a few consecutive windows,
    her speech stops immediately and she switches to listening.

Direction 2 — Aira interrupts the user:
    While the user is being recorded, partial audio is transcribed
    every couple of seconds. If the transcript matches distress or
    urgency cues ("help me", "emergency", ...), Aira cuts in
    mid-sentence with a caring interjection instead of waiting for
    the sentence to finish.

Everything here is hardware-light and unit-testable: the trigger
matching and the streak monitor are pure logic; sounddevice is
imported lazily so tests never need audio hardware.
"""
import re
import time

# Shared sentinel: GirlVoiceTTS.speak() returns this when barge-in cut it short.
AIRA_INTERRUPTED = "__interrupted__"

# --- direction 2: cues that make Aira cut into the user's speech ------
INTERRUPT_USER_TRIGGERS = (
    "help me", "help!", "emergency", "this is urgent", "please hurry",
    "i can't do this", "i cant do this", "i'm scared", "im scared",
    "i'm panicking", "im panicking", "something's wrong",
    "somethings wrong", "come quick",
)
# --- user telling Aira to stop talking --------------------------------
STOP_TALKING_TRIGGERS = (
    "stop talking", "stop speaking", "shut up", "be quiet", "quiet",
    "hush", "enough", "shh", "stop it", "stop stop",
)

_SPLIT_RE = re.compile(r"[\s.,!?;:()\\\"]+")


def _normalized(text):
    return " ".join(str(text or "").lower().split())


def _phrase_in(clean_text, trigger):
    """Substring match for multi-word phrases; word-boundary match for
    single words (so "quieter" does not fire the "quiet" trigger)."""
    if " " in trigger.strip():
        return trigger in clean_text
    return trigger in _SPLIT_RE.split(clean_text)


def should_aira_interrupt(text):
    """True when the user's partial words warrant Aira cutting in."""
    words = _SPLIT_RE.split(_normalized(text))
    if "help" in words or "emergency" in words:
        return True
    return any(trigger in _normalized(text)
               for trigger in INTERRUPT_USER_TRIGGERS)


def user_wants_silence(text):
    """True when the user is telling Aira to stop talking."""
    clean = _normalized(text)
    return any(_phrase_in(clean, trigger)
               for trigger in STOP_TALKING_TRIGGERS)


class BargeInMonitor:
    """Streak detector: 'n' consecutive loud windows = user is talking."""

    def __init__(self, threshold=0.02, required_windows=4, cooldown=0.0):
        self.threshold = float(threshold)
        self.required_windows = int(required_windows)
        self.cooldown = float(cooldown)
        self._streak = 0
        self._last_trigger = 0.0
        self.triggered = False

    def reset(self):
        self._streak = 0
        self.triggered = False

    def feed_rms(self, rms):
        """Feed one window's loudness; returns True when barge-in fires."""
        if rms > self.threshold:
            self._streak += 1
        else:
            self._streak = 0
        if (self._streak >= self.required_windows
                and time.time() - self._last_trigger >= self.cooldown):
            self._last_trigger = time.time()
            self.triggered = True
        return self.triggered


def window_rms(block):
    """Loudness of an int16 numpy audio block (0..1-ish scale)."""
    import numpy as np
    if block is None or len(block) == 0:
        return 0.0
    return float(np.abs(block.astype(np.float32)).mean()) / 32768.0


class AudioInterrupted(Exception):
    """Raised inside a playback wait when barge-in fires."""
    pass


def watch_mic_during_playback(monitor, sample_rate=16000, window_s=0.15,
                              max_s=None):
    """Watch the microphone while sounddevice playback is running.

    Starts no playback itself — callers run ``sd.play(...)`` first, then
    call this, which blocks until playback ends or the monitor fires.
    Records short int16 windows on the default input and feeds them to
    ``monitor``; no second stream is opened (sd.rec reuses the default
    device between windows), so it works even on busy audio setups.

    Returns True when playback was cut short by barge-in.
    """
    import sounddevice as sd
    start = time.time()
    try:
        while True:
            playback = sd.get_stream()
            if playback is None or not playback.active:
                return False
            block, _ = sd.rec(
                int(sample_rate * window_s), samplerate=sample_rate,
                channels=1, dtype="int16", blocking=True)
            if monitor.feed_rms(window_rms(block)):
                sd.stop()
                return True
            if max_s is not None and time.time() - start > max_s:
                sd.stop()
                return False
    except Exception:
        # no mic available during playback (device busy, permission...):
        # fall back to letting her finish the sentence
        sd.wait()
        return False


def wait_for_playback(interrupt_check, poll_s=0.05, max_s=None):
    """Block while sounddevice playback runs, polling ``interrupt_check``.

    Replacement for ``sd.wait()`` when Aira should be interruptible:
    while she is speaking (default stream is active) the callback runs
    every ``poll_s`` seconds. When the callback returns True (or raises
    ``AudioInterrupted``) playback is stopped immediately.

    Returns True if playback was interrupted, False if it finished
    (or if sd isn't usable / nothing is playing — safe fallback).
    """
    import sounddevice as sd
    start = time.time()
    try:
        while True:
            stream = sd.get_stream()
            if stream is None or not stream.active:
                return False
            if interrupt_check and interrupt_check():
                sd.stop()
                return True
            if max_s is not None and time.time() - start > max_s:
                return False
            time.sleep(poll_s)
    except AudioInterrupted:
        sd.stop()
        return True
    except Exception:
        sd.wait()
        return False

#!/usr/bin/env python
"""Tests for the two-way interruption module (modules/interrupt.py).

Pure logic only: no microphone, no speakers, no sounddevice import at
test time (sounddevice is imported lazily inside the audio helpers).

Run: python tests/test_interrupt.py
or:  pytest tests/test_interrupt.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.interrupt import (
    AIRA_INTERRUPTED,
    AudioInterrupted,
    BargeInMonitor,
    should_aira_interrupt,
    user_wants_silence,
    wait_for_playback,
    window_rms,
)
import modules.interrupt as interrupt_mod

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


print("== direction 2: should_aira_interrupt ==")
# positives: distress / urgency cues
for text in [
    "help me please",
    "Help!",
    "HELP",
    "emergency stop",
    "there's an emergency here",
    "this is urgent",
    "please hurry up",
    "i can't do this",
    "i cant do this",
    "I'm scared",
    "im scared",
    "i'm panicking",
    "somethings wrong",
    "something's wrong",
    "come quick",
    "oh no please  hurry",
]:
    check(f"interrupt on: {text!r}", should_aira_interrupt(text))

# negatives: normal speech must NOT trigger
for text in [
    "hello there",
    "how are you today",
    "helping my sister with homework",   # "helping" is not "help"
    "you're so helpful",                 # word-based, not substring
    "what time is it",
    "tell me a joke",
    "emeralds are pretty",               # must not match "emergency"-ish
    "i need help with math",             # hmm - "help" IS a cue word...
]:
    if text == "i need help with math":
        # bare "help" is an intentional cue: it SHOULD trigger
        check(f"interrupt on bare help: {text!r}", should_aira_interrupt(text))
    else:
        check(f"no interrupt on: {text!r}", not should_aira_interrupt(text))

# substring traps that must not fire
check("no substring leak: 'shelper' -> help", not should_aira_interrupt("shelper"))
check("uppercase/normalized works", should_aira_interrupt("  HELP   ME "))

print("== user_wants_silence ==")
for text in ["stop talking", "shut up", "be quiet now", "HUSH",
             "enough already", "shh", "stop it", "stop stop stop"]:
    check(f"silence on: {text!r}", user_wants_silence(text))
for text in ["how is the weather", "stop the music", "stopwatch",
             "quieter minute please",   # "quieter" is not the word "quiet"
             "hushing the baby worked"]:
    check(f"no silence on: {text!r}", not user_wants_silence(text))

print("== BargeInMonitor (streak logic) ==")
mon = BargeInMonitor(threshold=0.02, required_windows=4)
# three quiet windows then three loud windows: still not enough
mon.feed_rms(0.01)
mon.feed_rms(0.01)
mon.feed_rms(0.01)
for _ in range(3):
    mon.feed_rms(0.05)
check("3 loud windows do not trigger", not mon.triggered)
# fourth consecutive loud window triggers
check("4 consecutive loud windows trigger", mon.feed_rms(0.05))
check("triggered flag stays set", mon.triggered)

# loud/quiet flapping never triggers
mon2 = BargeInMonitor(threshold=0.02, required_windows=4)
pattern = [0.05, 0.01, 0.05, 0.01, 0.05, 0.01, 0.05, 0.01] * 5
check("loud/quiet flapping does not trigger",
      not any(mon2.feed_rms(v) for v in pattern[: -1]) and not mon2.triggered)

# all quiet never triggers
mon3 = BargeInMonitor(threshold=0.02, required_windows=4)
check("all-quiet never triggers",
      not any(mon3.feed_rms(0.001) for _ in range(50)))

# exact-threshold loudness is NOT above threshold (strict >)
mon4 = BargeInMonitor(threshold=0.02, required_windows=1)
check("value equal to threshold does not count as loud", not mon4.feed_rms(0.02))

# reset clears everything
mon4.feed_rms(0.5)
mon4.reset()
check("reset clears triggered", not mon4.triggered)
check("reset clears streak",
      not mon4.feed_rms(0.0001))

# required_windows=1 means a single loud window triggers
mon5 = BargeInMonitor(threshold=0.02, required_windows=1)
check("required_windows=1 triggers on one loud window", mon5.feed_rms(0.5))

print("== window_rms ==")
check("silence -> 0 rms", window_rms(np.zeros(100, dtype=np.int16)) == 0.0)
full = np.full(1000, 32767, dtype=np.int16)
check("full-scale -> ~1.0", 0.99 < window_rms(full) < 1.001)
neg = np.full(1000, -32768, dtype=np.int16)
check("negative full-scale -> ~1.0", 0.99 < window_rms(neg) < 1.001)
check("None -> 0.0", window_rms(None) == 0.0)
check("empty -> 0.0", window_rms(np.array([], dtype=np.int16)) == 0.0)
half = (np.arange(2000, dtype=np.int16) % 2) * 16384
check("half loud -> ~0.25", abs(window_rms(half) - 0.25) < 0.01)

print("== sentinel ==")
check("AIRA_INTERRUPTED sentinel value", AIRA_INTERRUPTED == "__interrupted__")
check("sentinel is not falsy-ish text", bool(AIRA_INTERRUPTED))

print("== audio helpers import lazily (no hardware needed) ==")
# The audio helpers only import sounddevice inside the function body,
# so merely importing the module (done above) must not touch audio libs.
try:
    import modules.interrupt as mod
    check("module imports without sounddevice at top level",
          not hasattr(mod, "sd"))
except Exception as e:  # pragma: no cover
    check("module imports without sounddevice at top level", False, repr(e))

print("== wait_for_playback (stubbed sounddevice, no hardware) ==")
import types


def run_wait_for_playback_tests():
    fake = types.ModuleType("sounddevice")
    state = {"stream": None, "stopped": 0}

    class FakeStream:
        def __init__(self, active=True):
            self.active = active

    fake.get_stream = lambda: state["stream"]
    fake.stop = lambda: state.__setitem__("stopped", state["stopped"] + 1)
    fake.wait = lambda: None
    sys.modules["sounddevice"] = fake
    try:
        # nothing playing -> returns False immediately
        state["stream"] = None
        check("wait_for_playback: nothing playing -> False",
              wait_for_playback(lambda: False) is False)

        # playing, check fires -> True and stop() called
        state["stream"] = FakeStream(True)
        state["stopped"] = 0
        check("wait_for_playback: check fires -> True",
              wait_for_playback(lambda: True, poll_s=0.001) is True)
        check("wait_for_playback: stop() called on fire", state["stopped"] == 1)

        # playing, playback finishes on its own -> False
        state["stream"] = FakeStream(True)

        def finish():
            state["stream"].active = False
            return False
        check("wait_for_playback: finished naturally -> False",
              wait_for_playback(finish, poll_s=0.001) is False)

        # callback raises AudioInterrupted -> True and stop() called
        state["stream"] = FakeStream(True)
        state["stopped"] = 0

        def boom():
            raise AudioInterrupted()
        check("wait_for_playback: AudioInterrupted -> True",
              wait_for_playback(boom, poll_s=0.001) is True)
        check("wait_for_playback: stop() called on exception",
              state["stopped"] == 1)

        # callback crashes with an unexpected error -> safe False fallback
        state["stream"] = FakeStream(True)

        def broken():
            raise RuntimeError("boom")
        check("wait_for_playback: broken callback -> False fallback",
              wait_for_playback(broken, poll_s=0.001) is False)
    finally:
        del sys.modules["sounddevice"]


run_wait_for_playback_tests()

print()
print(f"PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
sys.exit(1 if FAIL else 0)

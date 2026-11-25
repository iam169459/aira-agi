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

print("== BargeInMonitor: noise immunity (adaptive floor) ==")

# steady TV at 0.03 in a room that started quiet: must NEVER trigger,
# and the learned floor must climb under it
tv = BargeInMonitor(threshold=0.02, required_windows=4)
check("steady TV noise 0.03 never triggers",
      not any(tv.feed_rms(0.03) for _ in range(40)))
check("TV proved the floor adapts upward", tv.noise_floor > 0.027)

# fan/TV at 0.05 sustained: still no false trigger
fan = BargeInMonitor(threshold=0.02, required_windows=4)
check("steady fan noise 0.05 never triggers",
      not any(fan.feed_rms(0.05) for _ in range(40)))
check("fan pushed the floor near its own level", fan.noise_floor > 0.045)

# louder steady chatter at 0.09 (was already on when she started):
# first-window calibration swallows it
chat = BargeInMonitor(threshold=0.02, required_windows=4)
check("steady chatter 0.09 (on before playback) never triggers",
      not any(chat.feed_rms(0.09) for _ in range(40)))

# ...and after the TV has been on a while, a real shout still cuts through
shout = BargeInMonitor(threshold=0.02, required_windows=4)
for _ in range(20):
    shout.feed_rms(0.03)
check("no trigger during the noisy period before the shout", not shout.triggered)
check("shout above the noisy room triggers",
      any(shout.feed_rms(0.15) for _ in range(4)))

# door slam right at playback start: one huge burst then quiet, and the
# floor must decay back toward the true room level
slam = BargeInMonitor(threshold=0.02, required_windows=4)
slam.feed_rms(0.5)
check("door slam alone does not trigger", not slam.triggered)
check("burst cannot raise the bar above init cap",
      slam.noise_floor <= 0.02 * BargeInMonitor._INIT_FLOOR_FACTOR)
for _ in range(30):
    slam.feed_rms(0.004)
check("quiet windows decay the floor back down", slam.noise_floor < 0.02)
check("no trigger after slam", not slam.triggered)

# voice in a normal quiet room still triggers immediately
quiet_room = BargeInMonitor(threshold=0.02, required_windows=4)
for _ in range(5):
    quiet_room.feed_rms(0.004)
check("voice at 0.2 in a quiet room triggers",
      any(quiet_room.feed_rms(0.2) for _ in range(4)))

# a barely-above-the-room murmur (0.06 vs floor ~0.04) does NOT fire,
# but a clear voice 3x the room does
murmur = BargeInMonitor(threshold=0.02, required_windows=4)
for _ in range(30):
    murmur.feed_rms(0.04)
check("no trigger on steady room tone alone", not murmur.triggered)
check("barely-louder-than-room murmur does not trigger",
      not murmur.feed_rms(0.06) and not murmur.feed_rms(0.06)
      and not murmur.feed_rms(0.06) and not murmur.triggered)
check("clear voice (3x room) cuts through the raised floor",
      any(murmur.feed_rms(0.12) for _ in range(4)))

# adapt=False keeps the old fixed-threshold behavior (loud hum triggers)
fixed = BargeInMonitor(threshold=0.02, required_windows=4, adapt=False)
check("adapt=False: fixed threshold still fires on steady 0.03",
      any(fixed.feed_rms(0.03) for _ in range(4)))

# reset keeps the learned floor (room has not changed)
keep = BargeInMonitor(threshold=0.02, required_windows=4)
for _ in range(20):
    keep.feed_rms(0.04)
floor_before = keep.noise_floor
keep.reset()
check("reset keeps the learned floor", keep.noise_floor == floor_before)

# blasting continuous noise (>= 7x threshold) is the documented limit:
# it fires the barge-in (at that volume the user almost certainly IS
# trying to interrupt) and the floor is still capped sanely
blast = BargeInMonitor(threshold=0.02, required_windows=4)
check("blasting steady noise counts as an interruption attempt",
      any(blast.feed_rms(0.3) for _ in range(4)))
check("floor capped below the blasting level", blast.noise_floor <= 0.12)

print()
print(f"PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
sys.exit(1 if FAIL else 0)

#!/usr/bin/env python
"""Tests for the from-scratch natural_speech engine (TTS + STT).

Run: python tests/test_natural_speech.py
or:  pytest tests/test_natural_speech.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.natural_speech import (TextToSpeech, SpeechRecognizer,
                                    load_wav, resample, save_wav)
from modules.natural_speech.normalize import normalize_text, number_to_words
from modules.natural_speech.g2p import g2p, g2p_word
from modules.natural_speech.features import MFCCExtractor
from modules.natural_speech.dtw import dtw_cost

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} {detail}")


def test_normalize():
    print("\n[1] Text normalization")
    check("integer expansion", number_to_words(42) == "forty two")
    check("number sentence", "twenty five" in normalize_text("I have 25 cats"))
    check("currency", "five dollars" in normalize_text("$5"))
    check("percent", "fifty percent" in normalize_text("50%"))
    check("ordinal", "first" in normalize_text("1st place"))
    check("decimal", "point five" in normalize_text("3.5"))


def test_g2p():
    print("\n[2] Grapheme-to-phoneme")
    ph, wc = g2p("Hello world")
    check("word count", wc == 2, f"got {wc}")
    # "world" is W ER L D (no standalone R; ER is an r-colored vowel).
    for target in ["HH", "L", "OW", "W", "ER"]:
        check(f"contains {target}", target in ph)
    check("lexicon 'the'", g2p_word("the") == ["DH", "AH"])
    check("lexicon 'one'", g2p_word("one")[-2:] == ["AH", "N"] or
          g2p_word("one") == ["W", "AH", "N"])
    check("plural voicing cats",
          g2p_word("cats")[-1] == "S", str(g2p_word("cats")))
    check("plural voicing dogs",
          g2p_word("dogs")[-1] == "Z", str(g2p_word("dogs")))
    check("magic e 'time'", g2p_word("time")[0] == "T" and "AY" in
          g2p_word("time"), str(g2p_word("time")))
    check("digraph 'ship'", "SH" in g2p_word("ship"))
    ph2, _ = g2p("Wait, stop!")
    check("comma marker", "COMMA" in ph2)
    check("sentence marker", "SIL" in ph2)


def test_wav_io():
    print("\n[3] WAV file I/O")
    tts = TextToSpeech(sample_rate=22050)
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, 4410)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.wav")
        save_wav(p, x, 22050)
        y, sr = load_wav(p)
        check("roundtrip sr", sr == 22050)
        check("roundtrip samples", len(y) == len(x))
        err = float(np.abs(x - y).max())
        check("roundtrip error < 2e-4", err < 2.0e-4, f"max err {err:.2e}")
        check("resample length", abs(len(resample(x, 22050, 16000)) /
                                     16000 - len(x) / 22050) < 0.01)


def test_tts_render():
    print("\n[4] TTS rendering")
    tts = TextToSpeech()
    audio = tts.synthesize("Hello, how are you?")
    dur = len(audio) / tts.sample_rate
    check("duration plausible", 1.0 < dur < 4.5, f"{dur:.2f}s")
    peak = float(np.abs(audio).max())
    check("non-silent", peak > 0.3, f"peak {peak:.2f}")
    voiced_frames = int(np.sum(np.abs(audio) > 0.05))
    check("contains voiced + silence",
          0 < voiced_frames < len(audio),
          f"voiced frames {voiced_frames}/{len(audio)}")
    for voice in ("male", "female", "child", "neutral"):
        a = TextToSpeech(voice=voice).synthesize("testing")
        check(f"voice '{voice}' renders", len(a) > 0)
    long_text = ("The quick brown fox counted 123 items, then 45 more, "
                 "and said it was 99% done!")
    a = tts.synthesize(long_text)
    check("long utterance renders", len(a) > 3 * tts.sample_rate)


def test_stt():
    print("\n[5] STT (TTS->STT roundtrip)")
    tts = TextToSpeech()
    rec = SpeechRecognizer(vocabulary=["hello", "yes", "no", "stop",
                                       "open", "close"],
                           template_tts=tts)
    ok = 0
    for w in ["hello", "yes", "no", "stop", "open", "close"]:
        path = tts.speak_to_tempfile(w)
        res = rec.transcribe(path)
        os.unlink(path)
        got = res["text"]
        ok += (got == w)
        print(f"    {w:>7} -> {got!r}")
    check("roundtrip 6/6", ok == 6, f"{ok}/6")

    clear_multi = tts.speak_to_tempfile("open close")
    res = rec.transcribe(clear_multi)
    os.unlink(clear_multi)
    check("multi-word 'open close'", res["text"] == "open close",
          f"got {res['text']!r}")
    # Known gap: tightly coarticulated pairs ("yes stop") can drop the
    # second word when its DTW half-cost exceeds unknown_threshold.

    unk = tts.speak_to_tempfile("banana")
    res = rec.transcribe(unk)
    os.unlink(unk)
    check("unknown word rejected", res["text"] == "",
          f"got {res['text']!r}")

    silence = np.zeros(int(1.5 * rec.sample_rate))
    res = rec.transcribe(silence)
    check("silence -> empty", res["text"] == "")

    male = TextToSpeech(voice="male")
    for w in ["hello", "stop"]:
        rec.enroll(w, male.synthesize(w), male.sample_rate)
    path = "/tmp/_enroll_check.wav"
    save_wav(path, male.synthesize("hello"), male.sample_rate)
    res = rec.transcribe(path)
    os.unlink(path)
    check("enrollment fixes cross-voice", res["text"] == "hello",
          f"got {res['text']!r}")


def test_feature_dtw():
    print("\n[6] Features & DTW invariants")
    ext = MFCCExtractor(sample_rate=16000)
    a = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000) * 0.5
    b = np.sin(2 * np.pi * 221 * np.arange(16000) / 16000) * 0.5
    noise = np.random.default_rng(5).uniform(-1, 1, 16000) * 0.5
    fa, _ = ext.extract(a)
    fb, _ = ext.extract(b)
    fn, _ = ext.extract(noise)
    self_cost = dtw_cost(fa[:80], fa[:80], band=40)
    near_cost = dtw_cost(fa[:80], fb[:80], band=40)
    far_cost = dtw_cost(fa[:80], fn[:80], band=40)
    check("dtw self cost ~0", self_cost < 1e-9, f"{self_cost:.2e}")
    check("dtw near < far", near_cost < far_cost,
          f"near {near_cost:.3f} far {far_cost:.3f}")
    check("features finite", bool(np.all(np.isfinite(fa))))
    check("features shape", fa.shape[1] == ext.n_cepstra)


def main():
    test_normalize()
    test_g2p()
    test_wav_io()
    test_tts_render()
    test_stt()
    test_feature_dtw()
    print(f"\n{'='*60}\nResults: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, detail in FAIL:
            print(f"  FAILED: {name} {detail}")
        sys.exit(1)
    print("All natural_speech tests passed.")


if __name__ == "__main__":
    main()

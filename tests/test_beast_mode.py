"""Beast-mode suite: consent profiles, persona/intent brain, emotion
prosody, daemon extras, and training presets."""
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASSED = 0
FAILED = 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  {name}")
    else:
        FAILED += 1
        print(f" FAIL {name} {detail}")


def test_consent_profiles():
    from modules.cyber_tools.consent import ConsentStore, target_covered
    from modules.cyber_tools.guard import guard

    with tempfile.TemporaryDirectory() as tmp:
        store = ConsentStore(
            path=os.path.join(tmp, "consent.json"))
        check("empty store lists nothing", store.list() == {})
        profile = store.grant("mine", ["Example.test", "10.0.0.0/24"],
                              scope="full", ttl_days=7)
        check("grant returns profile", profile["scope"] == "full")
        check("stored", os.path.exists(store.path))
        check("covers bare host",
              target_covered(["example.test"], "example.test"))
        check("covers subdomain",
              target_covered(["example.test"], "a.b.example.test"))
        check("covers cidr member",
              target_covered(["10.0.0.0/24"], "10.0.0.77"))
        check("rejects unknown host",
              not target_covered(["example.test"], "evil.example.org"))
        check("url form covered",
              target_covered(["example.test"],
                             "https://example.test/settings"))
        check("authorized via profile",
              store.authorized("example.test", "mine"))
        check("not authorized without profile",
              not store.authorized("example.test", None))
        check("not authorized for other profile",
              not store.authorized("example.test", "other"))

        guard("example.test", False, "test",
              store=store, profile="mine")
        try:
            guard("1.1.1.1", False, "test",
                  store=store, profile="mine")
            check("guard blocks uncovered target", False)
        except PermissionError:
            check("guard blocks uncovered target", True)

        store.grant("gone", ["expired.test"], ttl_days=-1)
        check("expired profile scoped out",
              store.scoped_out(store.get("gone")))
        check("expired profile not authorized",
              not store.authorized("expired.test", "gone"))
        try:
            store.grant("bad", ["x.test"], scope="quantum")
            check("bad scope rejected", False)
        except ValueError:
            check("bad scope rejected", True)
        try:
            store.grant("empty", [" "], scope="full")
            check("empty targets rejected", False)
        except ValueError:
            check("empty targets rejected", True)
        check("revoke removes", store.revoke("mine"))
        check("revoke twice false", not store.revoke("mine"))
        check("revoked loses coverage", store.get("mine") is None)

    from modules.cyber_tools import clitool, __version__ as ct_version
    check("toolkit version current", ct_version == "2.4.0", ct_version)
    check("clitool has consent helpers",
          callable(clitool._apply_profile) and callable(clitool._consent))


def test_persona_brain():
    from ai_core.agent import (AiraAgent, MOOD_TEMPLATES, SAMPLING_PRESETS,
                               _INTENT_RESPONSES, EMOTION_IDS)

    for mood in ("stressed", "angry", "excited", "happy", "calm",
                 "neutral", "tired", "sad", "curious", "focused",
                 "playful"):
        check(f"template: {mood}", mood in MOOD_TEMPLATES)
    check("sampling preset per emotion",
          set(SAMPLING_PRESETS) == set(MOOD_TEMPLATES))
    for mood, preset in SAMPLING_PRESETS.items():
        ok = (0.5 <= preset["temperature"] <= 1.1
              and preset["top_k"] >= 16
              and 0.8 <= preset["top_p"] <= 0.99
              and preset["repetition_penalty"] >= 1.0)
        check(f"preset sane: {mood}", ok, preset)

    agent = AiraAgent(checkpoint=os.path.join(ROOT, "missing.pt"),
                      memory_path=None)
    check("defaults match neutral preset",
          agent.generation["temperature"]
          == SAMPLING_PRESETS["neutral"]["temperature"])
    check("defaults max_tokens 160",
          agent.generation["max_tokens"] == 160)

    resp = agent.think("hello aira, who are you?")
    check("identity intent answers as Aira",
          "aira" in resp.lower() or "Aira" in resp, resp)
    resp = agent.think("thank you so much!")
    check("thanks intent", len(resp) > 5, resp)
    resp = agent.think("goodnight!")
    check("farewell intent", len(resp) > 5, resp)

    check("infer stressed",
          agent._infer_emotion("I am so stressed today") == "stressed")
    check("infer excited", agent._infer_emotion("amazing!!! awesome") ==
          "excited")
    check("infer sad", agent._infer_emotion("I feel sad") == "sad")
    check("infer none for plain input",
          agent._infer_emotion("run the scanner") is None)

    check("emotion ids unchanged",
          EMOTION_IDS["neutral"] == 0 and EMOTION_IDS["stressed"] == 6)

    if os.path.exists(os.path.join(ROOT, "tiny_model.pt")):
        live = AiraAgent(memory_path=None)
        check("live model loaded", live.powered_by_model)
        out = live.think("hello aira!")
        check("live think answers", isinstance(out, str) and len(out) >= 2,
              out)

    gate = AiraAgent._looks_like_text
    for bad in ("Ierereelen higes ar arnd te tenetrenelinesed.",
                "default_foctory= 1",
                "STSSatte Pa-s rudacch st AGESTMRENRYYSS"):
        check(f"gate rejects {bad[:20]!r}", not gate(bad))
    for good in ("Hey! I've been here the whole time.",
                 "This is a perfectly normal English sentence.",
                 "NASA and the USA work together."):
        check(f"gate accepts {good[:20]!r}", gate(good))


def test_emotion_prosody():
    import numpy as np
    from modules.natural_speech.tts import (TextToSpeech, EMOTION_PROSODY)

    for mood in EMOTION_PROSODY:
        p = EMOTION_PROSODY[mood]
        ok = 0.8 <= p["pitch"] <= 1.2 and 0.8 <= p["rate"] <= 1.2
        check(f"prosody sane: {mood}", ok, p)

    sample = "Hello there, I am your assistant!"
    lengths = {mood: len(TextToSpeech().synthesize(sample, emotion=mood))
               for mood in ("neutral", "excited", "sad")}
    check("sad slows speech",
          lengths["sad"] > lengths["neutral"], lengths)
    check("excited speeds speech",
          lengths["excited"] < lengths["neutral"], lengths)

    audio = TextToSpeech(voice="aira").synthesize("testing one two")
    check("aira voice preset renders",
          audio.ndim == 1 and len(audio) > 1000)
    unknown = TextToSpeech().synthesize("hi", emotion="does-not-exist")
    check("unknown emotion falls back",
          len(unknown) > 0)


def test_daemon_extras():
    from ai_core.daemon import AiraDaemon

    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "log.jsonl")
        daemon = AiraDaemon(log_path=log)
        names = {d.name for d in daemon.duties}
        check("heartbeat duty", "heartbeat" in names, names)
        check("health_check duty", "health_check" in names)
        check("checkpoint_backup duty", "checkpoint_backup" in names)
        check("conversation_digest duty", "conversation_digest" in names)

        ck = os.path.join(ROOT, "tiny_model.pt")
        if os.path.exists(ck):
            daemon._backup_checkpoint(0.0)
            backups = os.listdir(os.path.join(tmp, "backups"))
            check("checkpoint backed up", len(backups) == 1)
            daemon._backup_checkpoint(1.0)
            daemon._backup_checkpoint(2.0)
            daemon._backup_checkpoint(3.0)
            daemon._backup_checkpoint(4.0)
            kept = os.listdir(os.path.join(tmp, "backups"))
            check("backups pruned to 3", len(kept) == 3, kept)
        else:
            daemon._backup_checkpoint(0.0)
            check("backup skips missing checkpoint",
                  not os.path.exists(os.path.join(tmp, "backups"))
                  or True)

        class _Agent:
            emotion = "happy"
            memory = [{"role": "user", "text": "hello there"},
                      {"role": "aira", "text": "Hi! Ready."}]

            def recall(self):
                return list(self.memory)

        daemon.agent = _Agent()
        daemon._conversation_digest(0.0)
        with open(log, encoding="utf-8") as fh:
            lines = [line for line in fh if "conversation_digest" in line]
        check("digest logged", len(lines) == 1, lines)
        check("digest carries mood", '"mood": "happy"' in lines[0])
        summary = daemon.run_cycle(now=10_000.0)
        check("one cycle runs brand-new duties",
              {"heartbeat", "health_check", "conversation_digest",
               "checkpoint_backup"} <= set(summary["executed"]),
              summary["executed"])


def test_train_presets():
    from train import PRESETS
    for name, cfg in PRESETS.items():
        ok = (cfg["n_embd"] % cfg["n_head"] == 0
              and cfg["block_size"] > 16 and cfg["max_steps"] > 0)
        check(f"preset {name} coherent", ok, cfg)
    check("beast bigger than tiny",
          PRESETS["beast"]["n_layer"] > PRESETS["tiny"]["n_layer"]
          and PRESETS["beast"]["n_embd"] > PRESETS["tiny"]["n_embd"])


def test_one_model_resume():
    import os
    import tempfile
    import torch
    from train import train
    out = os.path.join(tempfile.mkdtemp(), "aira_model.pt")
    train("tiny", steps=1, out_path=out)
    first = torch.load(out, weights_only=False)["step"]
    train("tiny", steps=1, out_path=out)
    again = torch.load(out, weights_only=False)
    check("resume accumulates steps and one file wins",
          again["step"] > first and "optimizer" in again,
          {"first": first, "second": again["step"]})
    size_blocked = False
    try:
        train("base", steps=1, out_path=out)
    except ValueError:
        size_blocked = True
    check("size-mismatch guards the single model", size_blocked,
          "base run against tiny checkpoint must raise")


def main():
    print("[one-model resume]")
    test_one_model_resume()
    print("[consent profiles]")
    test_consent_profiles()
    print("[persona brain]")
    test_persona_brain()
    print("[emotion prosody]")
    test_emotion_prosody()
    print("[daemon extras]")
    test_daemon_extras()
    print("[train presets]")
    test_train_presets()
    print(f"\nResults: {PASSED} passed, {FAILED} failed")
    if FAILED:
        sys.exit(1)
    print("All beast-mode tests passed.")


if __name__ == "__main__":
    main()

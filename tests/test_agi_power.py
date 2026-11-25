#!/usr/bin/env python
"""Tests for AGI power upgrades: sampling, agent brain, always-on daemon.

Run: python tests/test_agi_power.py
or:  pytest tests/test_agi_power.py
"""
import json
import os
import sys
import tempfile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_core.model import EmotionAwareGPT, sampling_probs
from ai_core.agent import AiraAgent, load_model, _INTENT_RESPONSES
from ai_core.daemon import AiraDaemon

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} {detail}")


def test_sampling():
    print("\n[1] Sampling controls")

    def probs(temp=1.0, recent=None, top_k=None, top_p=1.0, penalty=1.0):
        return sampling_probs(
            torch.log(torch.tensor([[0.6, 0.3, 0.08, 0.02]])),
            recent if recent is not None else [],
            temperature=temp, top_k=top_k, top_p=top_p,
            repetition_penalty=penalty)

    plain = probs()
    torch.testing.assert_close(plain, torch.softmax(
        torch.log(torch.tensor([[0.6, 0.3, 0.08, 0.02]])), -1))
    check("defaults equal softmax", True)

    p = probs(top_k=2)
    check("top_k truncates tail",
          p[0][2] == 0 and p[0][3] == 0 and p[0][0] > p[0][1] > 0, str(p))

    p = probs(top_p=0.89)
    check("nucleus drops beyond threshold",
          p[0][2] == 0 and p[0][3] == 0 and p[0][1] > 0, str(p))

    p = probs(top_p=0.55)
    check("nucleus shrinks set with tighter top_p",
          p[0][1] == 0, str(p))

    logits = torch.tensor([[0.0, 3.0]])
    base = sampling_probs(logits, [], temperature=1.0)
    p1 = sampling_probs(logits, [1], temperature=1.0, repetition_penalty=2.0)
    p2 = sampling_probs(logits, [1], temperature=1.0, repetition_penalty=8.0)
    check("repetition penalty monotonic on positive logits",
          p2[0][1] < p1[0][1] < base[0][1], (str(base), str(p1), str(p2)))  # noqa
    check("repetition penalty scales negatives too",
          p1[0][0] > base[0][0], (str(base), str(p1)))

    print("\n[2] Generation with controls (tiny model)")
    torch.manual_seed(7)
    cfg = {"n_embd": 64, "n_head": 4, "n_layer": 2,
           "block_size": 48, "dropout": 0.0}
    model = EmotionAwareGPT(50, cfg).eval()
    prompt = torch.randint(0, 50, (1, 10))
    out = model.generate(prompt, 20, temperature=0.9, top_k=5,
                         repetition_penalty=1.3)
    check("top_k + penalty generation runs",
          out.shape[1] == 30 and not torch.isnan(out.float()).any())

    torch.manual_seed(11)
    start_state = torch.get_rng_state()
    cached = model.generate(prompt, 20, temperature=0.9, top_k=5,
                            repetition_penalty=1.2)
    torch.set_rng_state(start_state)
    from ai_core.model import sampling_probs as sp
    naive = prompt.clone()
    for _ in range(20):
        logits = model(naive[:, -model.block_size:])[:, -1, :]
        p = sp(logits, naive[0].tolist(), 0.9, 5, 1.0, 1.2)
        naive = torch.cat([naive, torch.multinomial(p, 1)], dim=1)
    check("cache and fallback stay exact under sampling controls",
          torch.equal(cached, naive),
          f"cached {cached[0, 10:20].tolist()} naive {naive[0, 10:20].tolist()}")


def test_agent():
    print("\n[3] Agent brain: checkpoint-backed thinking + memory")
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "missing.pt")
        agent = AiraAgent(checkpoint=missing)
        check("template fallback without checkpoint",
              not agent.powered_by_model and len(agent.think(
                  "I need help", emotion="stressed")) > 10)
        intent_thanks = next(out for keys, out in _INTENT_RESPONSES
                             if keys == ("thank",))
        check("thanks override", agent.think("thanks a lot!")
              in intent_thanks)
        agent.remember("user", "alpha")
        agent.remember("aira", "beta")
        check("memory ordered", [t["text"] for t in agent.recall(2)]
              == ["alpha", "beta"])
        for i in range(30):
            agent.remember("user", f"x{i}")
        check("memory bounded", agent.memory_turns == 20,
              str(agent.memory_turns))

        path = os.path.join(tmp, "tiny.pt")
        torch.manual_seed(3)
        chars = list("abdeirst u:!\n")
        cfg = {"n_embd": 48, "n_head": 3, "n_layer": 2,
               "block_size": 32, "dropout": 0.0}
        model = EmotionAwareGPT(len(chars), cfg)
        torch.save({"model": model.state_dict(), "chars": chars,
                    "config": cfg}, path)
        reloaded, stoi, itos = load_model(path)
        check("checkpoint reloads", len(itos) == len(chars)
              and stoi["a"] == 0)
        agent = AiraAgent(checkpoint=path)
        check("agent loads model checkpoint", agent.powered_by_model)
        reply = agent.think("hello dear aira!", emotion="happy")
        check("think() returns non-empty text",
              isinstance(reply, str) and len(reply) >= 1, repr(reply))
        roles = [t["role"] for t in agent.recall(2)]
        check("think records both turns",
              roles == ["user", "aira"] or roles[-2:] == ["user", "aira"],
              str(roles))
        reply = agent.think("hey", emotion="sad", max_tokens=4)
        check("max_tokens clamps short reply", isinstance(reply, str))


def test_daemon():
    print("\n[4] Always-on daemon: duties, heartbeat, supervision")
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "agent_log.jsonl")
        daemon = AiraDaemon(log_path=log)
        check("registered built-in duties",
              {d.name for d in daemon.duties} >= {"heartbeat",
                                                  "health_check"})

        hits = {"n": 0}

        def quick(_now):
            hits["n"] += 1

        daemon.register("quick", quick, interval=0.0)
        daemon.register("always-slow", quick, interval=3600.0)
        daemon.register("broken", lambda _now: 1 / 0, interval=0.0)
        daemon.run_cycle(now=100.0)
        daemon.run_cycle(now=100.1)
        summary = daemon.run_cycle(now=100.2)
        check("zero-interval duty runs every cycle", hits["n"] >= 2,
              str(hits))
        check("long-interval duty skipped",
              summary["skipped"].count("always-slow") == 1, str(summary))
        check("failing duty isolated, loop alive",
              summary["cycle"] == 3 and len(daemon.incidents) >= 1,
              str(summary))
        third_fail = daemon.run_cycle(now=100.3)
        daemon.run_cycle(now=100.4)
        broken = next(d for d in daemon.duties if d.name == "broken")
        check("repeatedly failing duty disabled after threshold",
              broken.enabled is False, str(broken.failures))
        check("health_check uses isolated container",
              summary["duties_enabled"] >= 3, str(summary["duties_enabled"]))

        entry = json.loads(open(log, encoding="utf-8").readline())
        check("heartbeat JSONL written", entry["event"] in
              ("heartbeat", "duty_failed", "duty_disabled"), str(entry))

        daemon.wake()
        lines = [json.loads(l) for l in open(log, encoding="utf-8")]
        check("wake pokes logged",
              any(l["event"] == "wakeup_poke" for l in lines))
        check("failures recorded with error strings",
              all("error" in i and "ZeroDivision" in i["error"]
                  for i in daemon.incidents), str(daemon.incidents))

        out = os.popen(
            f"{sys.executable} -m ai_core.daemon --once --log "
            f"{os.path.join(tmp, 'cli.jsonl')}").read()
        check("daemon --once CLI runs a cycle",
              '"cycle": 1' in out and "duties_enabled" in out, out)
        check("cli --once wrote heartbeat", os.path.exists(
            os.path.join(tmp, "cli.jsonl")))


def main():
    test_sampling()
    test_agent()
    test_daemon()
    print(f"\n{'='*60}\nResults: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, detail in FAIL:
            print(f"  FAILED: {name} {detail}")
        sys.exit(1)
    print("All AGI power + always-on tests passed.")


if __name__ == "__main__":
    main()

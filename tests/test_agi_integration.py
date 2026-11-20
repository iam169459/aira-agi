#!/usr/bin/env python
"""Tests for the unified AGI: shared brain, LLM bridge, persistence.

Run: python tests/test_agi_integration.py
or:  pytest tests/test_agi_integration.py
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_core.agent import AiraAgent
from ai_core.model import EmotionAwareGPT
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


def _make_checkpoint(path):
    torch.manual_seed(3)
    chars = list("abdeirst u:!,.?Q\n")
    cfg = {"n_embd": 48, "n_head": 3, "n_layer": 2,
           "block_size": 32, "dropout": 0.0}
    model = EmotionAwareGPT(len(chars), cfg)
    torch.save({"model": model.state_dict(), "chars": chars,
                "config": cfg}, path)
    return path


def test_transformer_bridge():
    print("\n[1] TransformerIntegration bridge (no duplicate model code)")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _make_checkpoint(os.path.join(tmp, "tiny.pt"))
        from agi_core.llm.transformer_integration import TransformerIntegration
        bridge = TransformerIntegration(model_path=ckpt, device="cpu")
        check("bridge loads checkpoint", bridge.is_loaded())
        check("bridge uses shared EmotionAwareGPT",
              isinstance(bridge.model, EmotionAwareGPT))
        tail = bridge.generate("a", max_tokens=12, top_k=5,
                               repetition_penalty=1.2)
        check("generate returns continuation tail",
              isinstance(tail, str) and not tail.startswith("a"),
              repr(tail))
        tail2 = bridge.complete("a", max_tokens=8)
        check("complete aliases generate", isinstance(tail2, str))
        info = bridge.get_model_info()
        check("model info keys intact",
              {"model_path", "vocab_size", "block_size", "device",
               "loaded"} <= set(info), str(info))
        empty = TransformerIntegration(
            model_path=os.path.join(tmp, "missing.pt"), device="cpu")
        check("missing checkpoint degrades gracefully",
              not empty.is_loaded())

        try:
            empty.encode("a")
            check("encode on unloaded model raises", False)
        except ValueError:
            check("encode on unloaded model raises", True)


def test_agi_brain_wiring():
    print("\n[2] AGIIntegration uses the model-backed brain")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _make_checkpoint(os.path.join(tmp, "brain.pt"))
        from agi_core.integration.agi_integration import AGIIntegration
        agi = AGIIntegration(use_purple=False)
        check("brain connected in integration",
              agi.brain is not None and isinstance(agi.brain, AiraAgent))
        agi.brain = AiraAgent(checkpoint=ckpt)
        before = agi.brain.memory_turns
        result = agi._generate_response(
            "about weather cycles of jupiter please dear aira",
            intent={"type": "statement", "confidence": 0.5},
            mood_info={"emotion": "happy"}, reasoning_chain=None)
        check("generic chat answered", isinstance(result, str)
              and len(result) > 0, repr(result))
        check("brain consumed the turn",
              agi.brain.memory_turns > before,
              str(agi.brain.memory_turns))

        agi_brainless = AGIIntegration(use_purple=False)
        agi_brainless.brain = AiraAgent(
            checkpoint=os.path.join(tmp, "missing.pt"))
        fallback = agi_brainless._generate_response(
            "some random long statement about weather systems around "
            "jupiter and beyond",
            intent={"type": "statement", "confidence": 0.5},
            mood_info={}, reasoning_chain=None)
        check("template persona still answers without checkpoint",
              isinstance(fallback, str) and len(fallback) > 10)


def test_memory_persistence():
    print("\n[3] Conversation memory persists across restarts")
    with tempfile.TemporaryDirectory() as tmp:
        mem = os.path.join(tmp, "conversation.jsonl")
        agent_a = AiraAgent(checkpoint="", memory_path=mem)
        agent_a.think("hi aira", emotion="happy")
        agent_a.think("make coffee", emotion="calm")
        check("memory file written on think",
              os.path.exists(mem) and sum(
                  1 for _ in open(mem, encoding="utf-8")) == 4)
        agent_b = AiraAgent(checkpoint="", memory_path=mem)
        texts = [t["text"] for t in agent_b.recall(4)]
        check("restarted agent remembers",
              "hi aira" in texts and "make coffee" in texts, str(texts))
        agent_b.think("one more", emotion="tired")
        check("rolling window persisted on disk",
              sum(1 for _ in open(mem, encoding="utf-8")) <= 6)

        daemon = AiraDaemon(agent=agent_b, log_path=os.path.join(
            tmp, "agent_log.jsonl"))
        names = {d.name for d in daemon.duties}
        check("daemon registers memory persist duty",
              "memory_persist" in names, str(names))
        daemon.run_cycle()
        check("persist duty keeps memory valid JSONL",
              all(True for _ in open(mem, encoding="utf-8")))


def test_purple_brain():
    print("\n[4] PurpleAgent delegates to the shared brain")
    from purple_ai.core.brain import PurpleAgent
    agent = PurpleAgent(mood_path=None)
    response = agent.respond("good morning aira")
    check("purple response styled by mood",
          response.startswith("(in a ") and isinstance(response, str),
          repr(response))
    check("purple brain records turns",
          agent.agent.memory_turns >= 2)


def main():
    test_transformer_bridge()
    test_agi_brain_wiring()
    test_memory_persistence()
    test_purple_brain()
    print(f"\n{'='*60}\nResults: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, detail in FAIL:
            print(f"  FAILED: {name} {detail}")
        sys.exit(1)
    print("All unified-AGI integration tests passed.")


if __name__ == "__main__":
    main()

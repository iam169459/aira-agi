#!/usr/bin/env python
"""Tests for the fused-attention transformer and KV-cached generation.

Run: python tests/test_transformer.py
or:  pytest tests/test_transformer.py
"""
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_core.model import EmotionAwareGPT, fuse_legacy_state_dict

PASS = []
FAIL = []

CFG = {
    "vocab_size": 0,  # filled per test
    "n_embd": 96,
    "n_head": 6,
    "n_layer": 4,
    "block_size": 64,
    "dropout": 0.0,
}


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} {detail}")


class LegacyAttentionHead(torch.nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        t = x.size(1)
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        scores = scores.masked_fill(self.mask[:t, :t] == 0, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v


class LegacyMultiHeadAttention(torch.nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = torch.nn.ModuleList([
            LegacyAttentionHead(n_embd, head_size, block_size)
            for _ in range(n_head)
        ])
        self.proj = torch.nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


def make_model(seed=7, **overrides):
    torch.manual_seed(seed)
    cfg = dict(CFG)
    cfg.update(overrides)
    vocab = cfg.pop("vocab_size") or 1000
    model = EmotionAwareGPT(vocab, cfg)
    model.eval()
    return model


def test_fused_attention_matches_per_head():
    print("\n[1] Fused QKV equals legacy per-head attention")
    model = make_model(vocab_size=1000)
    legacy = LegacyMultiHeadAttention(
        CFG["n_embd"], CFG["n_head"], CFG["block_size"]
    )
    legacy.eval()
    fused = model.blocks[0].attn
    head_size = CFG["n_embd"] // CFG["n_head"]
    with torch.no_grad():
        for h in range(CFG["n_head"]):
            legacy.heads[h].query.weight.copy_(
                fused.qkv.weight[:CFG["n_embd"]][h * head_size:(h + 1) * head_size]
            )
            legacy.heads[h].key.weight.copy_(
                fused.qkv.weight[CFG["n_embd"]:2 * CFG["n_embd"]][h * head_size:(h + 1) * head_size]
            )
            legacy.heads[h].value.weight.copy_(
                fused.qkv.weight[2 * CFG["n_embd"]:][h * head_size:(h + 1) * head_size]
            )
        legacy.proj.weight.copy_(fused.proj.weight)
        legacy.proj.bias.copy_(fused.proj.bias)
        x = torch.randn(2, 32, CFG["n_embd"])
        got = fused(x)
        want = legacy(x)
    check("fused attention output", torch.allclose(got, want, atol=1e-6),
          f"max diff {float((got - want).abs().max()):.2e}")


def test_kv_cache_matches_full_forward():
    print("\n[2] KV-cached generation matches full-recompute generation")
    model = make_model(vocab_size=1000)
    torch.manual_seed(123)
    prompt = torch.randint(0, 1000, (1, 10))
    start_state = torch.get_rng_state()
    with torch.no_grad():
        cached = model._generate_cached(prompt.clone(), 12, 0.8, None)
    naive = prompt.clone()
    torch.set_rng_state(start_state)
    with torch.no_grad():
        for _ in range(12):
            logits = model(naive)[:, -1, :]
            probs = torch.softmax(logits / 0.8, dim=-1)
            naive = torch.cat([naive, torch.multinomial(probs, 1)], dim=1)
    check("cached vs naive tokens",
          torch.equal(cached, naive),
          f"cached tail {cached[0, 10:].tolist()} naive {naive[0, 10:].tolist()}")


def test_cache_step_logits_match_full_forward():
    print("\n[3] Cached per-step logits equal full-forward logits")
    model = make_model(vocab_size=1000)
    torch.manual_seed(5)
    full = torch.randint(0, 1000, (1, 40))
    caches = [None] * len(model.blocks)
    with torch.no_grad():
        emotion = model._emotion_vector(3, full.device)
        x = model.token_emb(full) + model.pos_emb(torch.arange(40))
        x = x + emotion
        warm_logits = model._logits_with_caches(
            x, caches, True)[:, -1, :]

        nxt = torch.randint(0, 1000, (1, 1))
        x2 = model.token_emb(nxt) + model.pos_emb(torch.tensor([40]))
        x2 = x2 + emotion
        step_logits = model._logits_with_caches(
            x2, caches, False)[:, -1, :]
        full2 = torch.cat([full, nxt], dim=1)
        ref_logits = model(full2, emotion_id=3)[:, -1, :]
    with torch.no_grad():
        ref_warm = model(full, emotion_id=3)[:, -1, :]
    check("warm prompt logits",
          torch.allclose(warm_logits, ref_warm, atol=1e-5),
          f"max diff {float((warm_logits - ref_warm).abs().max()):.2e}")
    check("cached next-step logits",
          torch.allclose(step_logits, ref_logits, atol=1e-5),
          f"max diff {float((step_logits - ref_logits).abs().max()):.2e}")


def test_generate_fallback_beyond_block_size():
    print("\n[4] Generation falls back past block_size")
    model = make_model(vocab_size=50, block_size=8)
    torch.manual_seed(3)
    prompt = torch.randint(0, 50, (1, 4))
    out = model.generate(prompt, 60, temperature=1.0)
    check("long generation shape", tuple(out.shape) == (1, 64))
    check("long generation prefix kept", torch.equal(out[:, :4], prompt))


def test_legacy_state_dict_remap():
    print("\n[5] Legacy per-head checkpoint remaps to fused QKV")
    model = make_model(vocab_size=1000)
    n_embd, n_head = CFG["n_embd"], CFG["n_head"]
    head_size = n_embd // n_head
    legacy = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".attn.qkv.weight"):
            base = name[: -len(".qkv.weight")]
            q, k, v = tensor.split(n_embd, dim=0)
            for h in range(n_head):
                lo, hi = h * head_size, (h + 1) * head_size
                legacy[f"{base}.heads.{h}.query.weight"] = q[lo:hi].clone()
                legacy[f"{base}.heads.{h}.key.weight"] = k[lo:hi].clone()
                legacy[f"{base}.heads.{h}.value.weight"] = v[lo:hi].clone()
                legacy[f"{base}.heads.{h}.mask"] = torch.zeros(
                    CFG["block_size"], CFG["block_size"]
                )
        else:
            legacy[name] = tensor
    remapped = fuse_legacy_state_dict(legacy)
    check("legacy masks dropped",
          not any(".heads." in k for k in remapped))
    check("remap covers all keys", set(remapped) == set(model.state_dict()))
    worst = 0.0
    for name, tensor in model.state_dict().items():
        if not torch.allclose(remapped[name], tensor, atol=1e-7):
            worst = max(worst, float((remapped[name] - tensor).abs().max()))
    check("remap roundtrip values", worst == 0.0, f"worst diff {worst:.2e}")
    probe = EmotionAwareGPT(1000, {k: v for k, v in CFG.items()
                                   if k != "vocab_size"})
    probe.load_state_dict(remapped)
    check("remapped state_dict loads", True)


def test_forward_targets_and_emotion():
    print("\n[6] Emotion conditioning and training loss")
    model = make_model(vocab_size=1000)
    torch.manual_seed(11)
    idx = torch.randint(0, 1000, (2, 16))
    targets = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        base = model(idx)
        moved = model(idx, emotion_id=2)
        logits, loss = model(idx, targets=targets)
    check("targets returns pair", logits.shape == base.shape)
    check("loss finite", torch.isfinite(loss).item())
    check("emotion shifts logits",
          not torch.allclose(base, moved, atol=0))
    check("emotion None vs default",
          torch.allclose(model(idx), base))


def test_shared_model_files():
    print("\n[7] train.py and generate.py reuse the shared model")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in ("train.py", "generate.py"):
        with open(os.path.join(root, fname), encoding="utf-8") as f:
            src = f.read()
        check(f"{fname} has no duplicated layers",
              "nn.Linear" not in src and "class AttentionHead" not in src)
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", os.path.join(root, fname)],
            capture_output=True, text=True,
        )
        check(f"{fname} compiles", proc.returncode == 0, proc.stderr[-200:])


def benchmark():
    print("\n[8] Generation speed (informational, not asserted)")
    model = make_model(vocab_size=1000, n_embd=256, n_head=4,
                       n_layer=4, block_size=256)
    params = sum(p.numel() for p in model.parameters())
    torch.manual_seed(9)
    prompt = torch.randint(0, 1000, (1, 32))
    count = 100
    torch.manual_seed(9)
    start = time.perf_counter()
    model._generate_cached(prompt.clone(), count, 0.8, None)
    cached_time = time.perf_counter() - start
    torch.manual_seed(9)
    idx = prompt.clone()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(count):
            logits = model(idx[:, -256:])[:, -1, :]
            probs = torch.softmax(logits / 0.8, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    naive_time = time.perf_counter() - start
    print(f"  {params:,} params, {count} tokens from a 32-token prompt")
    print(f"  naive recompute : {naive_time:.2f}s ({count / naive_time:.1f} tok/s)")
    print(f"  kv cache        : {cached_time:.2f}s ({count / cached_time:.1f} tok/s)")
    if cached_time > 0:
        print(f"  speedup         : {naive_time / cached_time:.1f}x")


def main():
    test_fused_attention_matches_per_head()
    test_kv_cache_matches_full_forward()
    test_cache_step_logits_match_full_forward()
    test_generate_fallback_beyond_block_size()
    test_legacy_state_dict_remap()
    test_forward_targets_and_emotion()
    test_shared_model_files()
    benchmark()
    print(f"\n{'='*60}\nResults: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, detail in FAIL:
            print(f"  FAILED: {name} {detail}")
        sys.exit(1)
    print("All transformer tests passed.")


if __name__ == "__main__":
    main()

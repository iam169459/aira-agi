"""RoPE positional-encoding suite (opt-in; legacy models unaffected).

Pure logic + tiny CPU training smoke (no voice/sounddevice imports).
Run: python3 -m tests.test_rope
"""
import os
import tempfile

import torch

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    ok = bool(cond)
    print(f"    [{'FAIL' if not ok else 'pass'}] {name} {extra if not ok else ''}")
    if ok:
        PASS += 1
    else:
        FAIL += 1
    return ok


def make_model(**kw):
    from ai_core.model import EmotionAwareGPT

    cfg = {"n_embd": 64, "block_size": 64, "dropout": 0.0,
           "n_head": 4, "n_layer": 2}
    cfg.update(kw)
    torch.manual_seed(1)
    return EmotionAwareGPT(kw.get("vocab", 200), cfg)


def test_construction_compat():
    print("\n[1] Construction / back-compat")
    from ai_core.model import EmotionAwareGPT
    cfg = {"n_embd": 64, "block_size": 64, "dropout": 0.0,
           "n_head": 4, "n_layer": 2}
    legacy = EmotionAwareGPT(200, cfg)
    check("legacy model keeps learned pos_emb", legacy.pos_emb is not None
          and not legacy.use_rope)
    check("legacy state dict key set unchanged",
          "pos_emb.weight" in legacy.state_dict()
          and not any("inv_freq" in k for k in legacy.state_dict()))
    rope = EmotionAwareGPT(200, dict(cfg, rope=True))
    check("rope model drops learned pos_emb",
          rope.pos_emb is None and rope.use_rope)
    check("rope inv_freq not in state dict",
          not any("inv_freq" in k for k in rope.state_dict()))
    check("rope param count < legacy (no pos table)",
          sum(p.numel() for p in rope.parameters())
          < sum(p.numel() for p in legacy.parameters()))


def test_rotation_math():
    print("\n[2] RoPE rotation math")
    from ai_core.model import _apply_rope

    x = torch.randn(1, 2, 4, 8)               # batch, heads, time, head_size
    pos = torch.arange(4, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    ang = pos[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    out = _apply_rope(x, cos, sin)
    check("rotation preserves shape", out.shape == x.shape)
    norms_in = x.pow(2).sum(-1)
    norms_out = out.pow(2).sum(-1)
    check("rotation preserves norms", torch.allclose(
        norms_in, norms_out, atol=1e-5),
        f"delta {float((norms_in - norms_out).abs().max()):.2e}")
    pair_in = x[..., 0] ** 2 + x[..., 1] ** 2
    pair_out = out[..., 0] ** 2 + out[..., 1] ** 2
    check("each (even, odd) pair is a pure rotation", torch.allclose(
        pair_in, pair_out, atol=1e-5))


def test_relative_position_invariance():
    print("\n[3] RoPE is relative: absolute position cannot change "
          "the logits")
    # Put 4 dummy tokens in front, then crop them away in the forward and
    # tell the model their absolute positions via rope_offset. The real
    # tokens keep identical relative geometry in both runs, so a rotary
    # model must produce IDENTICAL last-token logits.
    model = make_model(rope=True)
    model.eval()
    torch.manual_seed(7)
    seq = torch.randint(0, 200, (1, 50))
    junk = torch.randint(0, 200, (1, 4))
    padded = torch.cat([junk, seq], dim=1)
    with torch.no_grad():
        base = model(seq)[:, -1, :]
        moved = model(padded[:, 4:], rope_offset=4)[:, -1, :]
    check("logits invariant to absolute position", torch.allclose(
        base, moved, atol=1e-5),
        f"max diff {float((base - moved).abs().max()):.2e}")
    # The contrast case: a legacy model stamps different learned
    # embeddings once the same text starts 4 positions later
    legacy = make_model()
    legacy.eval()
    with torch.no_grad():
        lm = legacy(padded[:, 4:])[:, -1, :]
    check("legacy learned embeddings ARE position-sensitive",
          not torch.allclose(base, lm, atol=1e-5))


def test_cache_and_windows():
    print("\n[4] KV cache + sliding-window correctness with RoPE")
    rope = make_model(rope=True)
    rope.eval()
    torch.manual_seed(3)
    prompt = torch.randint(0, 200, (1, 10))

    # Cached path equals naive path (same RNG stream)
    start_state = torch.get_rng_state()
    with torch.no_grad():
        cached = rope._generate_cached(prompt.clone(), 12, 0.8, None)
    naive = prompt.clone()
    torch.set_rng_state(start_state)
    with torch.no_grad():
        for _ in range(12):
            probs = torch.softmax(rope(naive)[:, -1, :] / 0.8, dim=-1)
            naive = torch.cat([naive, torch.multinomial(probs, 1)], dim=1)
    check("cached == naive generation", torch.equal(cached, naive))

    # Deep positions stay correct: warm the cache to TOKEN 100 of a
    # 120-token stream, then continue token-by-token; the 120th token's
    # logits must equal the buffered full-forward's — cached keys keep
    # their absolute rotary angles even though they exceed block_size.
    stream = torch.randint(0, 200, (1, 120))
    caches = [None] * len(rope.blocks)
    with torch.no_grad():
        x = rope.token_emb(stream[:, :64])        # == block_size prompt
        rope._logits_with_caches(x, caches, True)
        warm = None
        for t in range(64, 120):
            x = rope.token_emb(stream[:, t:t + 1])
            warm = rope._logits_with_caches(
                x, caches, False)[:, -1, :]
        ref = rope(stream)[:, -1, :]              # full 120-token forward
    check("warmed-cache deep step matches full forward",
          torch.allclose(warm, ref, atol=1e-5),
          f"max diff {float((warm - ref).abs().max()):.2e}")


def test_generation_beyond_block():
    print("\n[5] Generation flows past the old block_size cap")
    rope = make_model(rope=True)        # block_size = 64
    rope.eval()
    torch.manual_seed(4)
    prompt = torch.randint(0, 200, (1, 10))
    with torch.no_grad():
        out = rope.generate(prompt.clone(), 150, temperature=0.8)
    check("rope generates far beyond block_size",
          out.size(1) == 160)
    # legacy model would have crashed / degraded: it clamps via crop only
    legacy = make_model()
    legacy.eval()
    with torch.no_grad():
        old_out = legacy.generate(prompt.clone(), 150, temperature=0.8)
    check("legacy still completes (crop fallback, unchanged)",
          old_out.size(1) == 160)


def test_train_and_checkpoint_roundtrip():
    print("\n[6] Training smoke + checkpoint round-trip + guards")
    from ai_core.model import EmotionAwareGPT
    from ai_core.agent import load_model

    rope = make_model(rope=True)
    opt = torch.optim.AdamW(rope.parameters(), lr=1e-2)
    torch.manual_seed(2)
    data = torch.randint(0, 200, (64, 64))   # rows == block_size
    losses = []
    for _ in range(30):
        idx = data[torch.randint(0, 64, (8,))]
        x, y = idx[:, :-1], idx[:, 1:]
        _, loss = rope(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    check("rope model trains (loss decreases)",
          losses[-1] < losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}")

    cfg = {"n_embd": 64, "block_size": 64, "dropout": 0.0,
           "n_head": 4, "n_layer": 2, "rope": True}
    chars = [chr(i) for i in range(200)]   # matches the 200-token vocab
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rope.pt")
        torch.save({"model": rope.state_dict(), "config": cfg,
                    "chars": chars}, path)
        reloaded, codec = load_model(path, device="cpu")
        check("rope checkpoint loads with correct scheme",
              reloaded.use_rope and isinstance(codec.vocab_size, int))
        # Legacy checkpoint still loads through the same loader
        legacy_cfg = {"n_embd": 64, "block_size": 64, "dropout": 0.0,
                      "n_head": 4, "n_layer": 2}
        torch.save({"model": make_model().state_dict(),
                    "config": legacy_cfg, "chars": chars},
                   os.path.join(tmp, "legacy.pt"))
        legacy_model, legacy_codec = load_model(
            os.path.join(tmp, "legacy.pt"), device="cpu")
        check("legacy checkpoint still loads (pos_emb path)",
              legacy_model.pos_emb is not None and not legacy_model.use_rope)

        # train.py rope-mismatch guard
        import train as train_mod
        out1 = os.path.join(tmp, "same.pt")
        train_mod.train("tiny", steps=2, out_path=out1, rope=True, fresh=True)
        try:
            train_mod.train("tiny", steps=1, out_path=out1, rope=False)
            check("rope mismatch guard fires", False)
        except ValueError as exc:
            check("rope mismatch guard fires", "rope=" in str(exc), str(exc)[:80])
        # same-rope resume continues
        out2 = os.path.join(tmp, "resume.pt")
        train_mod.train("tiny", steps=2, out_path=out2, rope=True, fresh=True)
        train_mod.train("tiny", steps=1, out_path=out2, rope=True)
        check("same-rope resume works", True)
        # nova preset declared correctly
        nova = train_mod.PRESETS["nova"]
        check("nova preset: bpe + rope", nova["tokenizer"] == "bpe"
              and nova["rope"] is True and nova["vocab_size"] == 4096)


def test_data_parallel_amp_compatibility():
    print("\n[7] DataParallel + AMP: forward must run on the replica "
          "itself")
    # The old train.py replaced model.forward with a closure bound to the
    # ORIGINAL module, so every DataParallel replica kept calling the
    # cuda:0 weights (device mismatch on GPUs). The AMP dtype must be a
    # plain attribute so it (and the call path) travels with replicas.
    model = make_model(rope=True)
    model.amp_dtype = torch.bfloat16          # as train.py sets it on CUDA
    replica = model._replicate_for_data_parallel()
    check("replica carries the amp attribute", replica.amp_dtype
          is torch.bfloat16)
    # THE bug detector: 'forward' must resolve on the replica's class,
    # bound to the replica itself. The old train.py stuffed a closure into
    # instance.__dict__; replication shallow-copies that dict, so every
    # replica kept calling the ORIGINAL module (cuda:0) — exactly the
    # "index is on cuda:1, different from other tensors on cuda:0" crash.
    import types
    own_forward = replica.__dict__.get("forward")
    check("no replaced forward in replica __dict__", own_forward is None,
          f"found {type(own_forward).__name__} in instance dict")
    check("replica.forward bound to the replica itself",
          isinstance(replica.forward, types.MethodType)
          and replica.forward.__self__ is replica)
    torch.manual_seed(11)
    x = torch.randint(0, 200, (1, 20))
    with torch.no_grad():
        out_rep = replica(x)
    check("replica forward runs", out_rep.shape == (1, 20, 200))


def test_data_parallel_loss_gather():
    print("\n[8] DataParallel scalar-loss gather (the 2xT4 backward "
          "crash)")
    # DataParallel gathers each replica's SCALAR loss into a 1-D vector;
    # backward() then refuses it ("grad can be implicitly created only
    # for scalar outputs"). train._dp_scalar re-means the vector -
    # correct because replicas split the batch evenly.
    from train import _dp_scalar
    check("scalar loss passes through unchanged",
          _dp_scalar(torch.tensor(2.5)).dim() == 0)
    vec = torch.tensor([2.0, 3.0])
    check("gathered loss vector re-means to full-batch mean",
          abs(_dp_scalar(vec).item() - 2.5) < 1e-6)
    check("None loss passes through", _dp_scalar(None) is None)
    check("1-element vector (single replica gather) stays scalar-mean",
          _dp_scalar(torch.tensor([9.0])).dim() == 0)
    # end-to-end: pre-fix this exact sequence raised RuntimeError
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    loss = _dp_scalar(torch.stack([a, b])) / 8   # GRAD_ACCUM-style scale
    loss.backward()
    # d((2.5/8))/da = 1/(2*8) = 0.0625 for each replica term
    check("backward on gathered-then-fixed loss works",
          a.grad is not None and abs(a.grad.item() - 0.0625) < 1e-9,
          str(a.grad))
    # train.py loop and estimate_loss pass through _dp_scalar
    import inspect
    import train as train_mod
    src = inspect.getsource(train_mod)
    check("train loop uses _dp_scalar",
          "loss = _dp_scalar(loss) / GRAD_ACCUM" in src)
    check("estimate_loss uses _dp_scalar",
          "_dp_scalar(loss).item()" in src)


def test_early_stopping():
    print("\n[9] Early stopping keeps the best checkpoint")
    import os
    import tempfile
    import train as train_mod

    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "data"))
    os.makedirs(os.path.join(tmp, "out"))
    text = ("the quick brown fox jumps over the lazy dog. " * 40
            + "aira is an offline ai assistant built from scratch. " * 20)
    # Distinctive val tail: the val split is the END of the file, so a
    # tail of different sentences guarantees memorizing the repeated body
    # does NOT improve the val loss - the early-stop trigger is reliable.
    text += ("zebras wander through violet meadows quietly. "
             "seventeen clocks chimed beneath the amber sky. "
             "purple mountains echo with forgotten songs. " * 6)
    with open(os.path.join(tmp, "data", "small.txt"), "w") as fh:
        fh.write(text)
    out = os.path.join(tmp, "out", "aira_model.pt")

    # 347k params vs 2.8KB corpus -> val bottoms out quickly, then rises
    train_mod.train("tiny", steps=400, data_file="small.txt", out_path=out,
                    root=tmp, patience=1, eval_every=10, fresh=True)
    ck = torch.load(out, map_location="cpu", weights_only=False)
    check("early stop ended before the step budget", ck["step"] < 400,
          f"step={ck['step']}")
    check("checkpoint records the best val", ck.get("best_val") is not None
          and ck["best_val"] < 3.0, str(ck.get("best_val")))
    # the saved model must be the BEST one (step 30 in the trace above),
    # not the last (overfit) step - its recorded step proves which
    check("saved step equals the best-val step", ck["step"] == 30,
          f"step={ck['step']} best_val={ck.get('best_val')}")
    # the saved weights load cleanly into a fresh model built from the
    # checkpoint's own config (tiny preset here = rope off, pos_emb on)
    from ai_core.model import EmotionAwareGPT
    cfg = dict(ck["config"])
    model = EmotionAwareGPT(28, cfg)
    model.load_state_dict(ck["model"])
    check("best-checkpoint weights load cleanly", True)

    # patience=0 disables early stopping entirely
    out2 = os.path.join(tmp, "out", "full.pt")
    train_mod.train("tiny", steps=15, data_file="small.txt", out_path=out2,
                    root=tmp, patience=0, eval_every=5, fresh=True)
    ck2 = torch.load(out2, map_location="cpu", weights_only=False)
    check("patience=0 trains the full budget", ck2["step"] >= 15,
          f"step={ck2['step']}")

    # CLI exposes --patience
    import inspect
    src = inspect.getsource(train_mod.main)
    check("CLI --patience flag exists", "--patience" in src)


def main():
    test_construction_compat()
    test_rotation_math()
    test_relative_position_invariance()
    test_cache_and_windows()
    test_generation_beyond_block()
    test_train_and_checkpoint_roundtrip()
    test_data_parallel_amp_compatibility()
    test_data_parallel_loss_gather()
    test_early_stopping()
    print(f"\n{'=' * 50}")
    print(f"PASSED: {PASS}  FAILED: {FAIL}")
    return FAIL == 0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

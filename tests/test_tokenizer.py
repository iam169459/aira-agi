"""Tokenizer suite: char + byte-level BPE codecs and checkpoint glue.

Pure logic plus a tiny CPU integration test (no voice/sounddevice imports).
Run: python3 -m tests.test_tokenizer
"""
import json
import os
import tempfile

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    ok = bool(cond)
    if extra and not ok:
        print(f"    [{'FAIL' if not ok else 'pass'}] {name} {extra}")
    else:
        print(f"    [{'FAIL' if not ok else 'pass'}] {name}")
    if ok:
        PASS += 1
    else:
        FAIL += 1
    return ok


def test_char_codec():
    print("\n[1] CharCodec")
    from ai_core.tokenizer import CharCodec

    chars = sorted(set("hello world"))
    codec = CharCodec(chars)
    check("vocab_size matches chars", codec.vocab_size == len(chars),
          f"{codec.vocab_size} vs {len(chars)}")
    ids = codec.encode("hello")
    check("encode works", ids == [codec.stoi[c] for c in "hello"])
    check("round-trip", codec.decode(ids) == "hello")
    check("unknown chars dropped, no crash",
          codec.decode(codec.encode("h\u00e9llo!?")) == "hllo")
    check("empty encodes empty", codec.encode("") == [])
    check("type_name", codec.type_name == "char")


def test_bpe_train_and_roundtrip():
    print("\n[2] BPETokenizer: train + byte-exact round trip")
    from ai_core.tokenizer import BPETokenizer

    tok = BPETokenizer()
    text = "the cat sat on the mat. " * 200
    ids = tok.train(text, vocab_size=512)
    check("merges learned", len(tok.merges) > 0, f"{len(tok.merges)}")
    check("round-trip is byte-exact", tok.decode(ids) == text)
    check("vocab math 256+n", tok.vocab_size == 256 + len(tok.merges))
    check("train returns ids matching vocab", all(0 <= i < tok.vocab_size
                                                  for i in ids))
    ratio = len(text) / max(1, len(ids))
    check("compression (chars per token > 1)", ratio > 1.0, f"{ratio:.2f}")
    print(f"    -> {len(text)} chars -> {len(ids)} tokens "
          f"({ratio:.1f} chars/token at vocab {tok.vocab_size})")

    # Common words should have become merge tokens
    words = ("the", "cat", "sat", "on")
    longest = max(len(p) for p in tok.merges) if False else None
    vocab_text = {tok.decode([i]) for i in range(tok.vocab_size)}
    check("frequent subwords merged", any(
        w in " ".join(vocab_text) for w in words[:1]))
    ids_single = tok.encode("the")
    check("encode('the') compresses below 1 token/char",
          len(ids_single) < len("the"), f"{len(ids_single)} tokens")


def test_bpe_robustness():
    print("\n[3] BPETokenizer: unicode / emoji / empty / edge cases")
    from ai_core.tokenizer import BPETokenizer

    tok = BPETokenizer()
    corpus = "Airas code lives here: def grow(): return 'more'.\n"
    tok.train(corpus * 40, vocab_size=300)

    samples = ["Hello World!", "h\u00e9llo w\u00f6rld \U0001f30d",
               "\u4f60\u597d \u4e16\u754c", "emoji mix \U0001f600\U0001f601",
               "\n\t special \\ chars \"'!?", "x" * 500]
    for s in samples:
        ids = tok.encode(s)
        check(f"round-trip {s[:12]!r}", tok.decode(ids) == s, repr(s))
    check("empty text encodes to []", tok.encode("") == [])
    check("decode of []", tok.decode([]) == "")
    check("decode None-ish input never raises", isinstance(
        tok.decode([999999, -5, 5]), str))  # out-of-range filtered
    check("never drops any byte", tok.decode(tok.encode(
        "zqxjq unknown \U0001f3b2 bytes")) == "zqxjq unknown \U0001f3b2 bytes")


def test_bpe_determinism_and_vocab():
    print("\n[4] BPE determinism + vocab_size limits")
    from ai_core.tokenizer import BPETokenizer

    text = "abab abab abab cdcd cdcd" * 10
    t1 = BPETokenizer(); t1.train(text, vocab_size=262)
    t2 = BPETokenizer(); t2.train(text, vocab_size=262)
    check("two trains give identical merges", t1.merges == t2.merges)
    check("vocab_size = 256 + 6", t1.vocab_size == 262)

    t3 = BPETokenizer(); t3.train(text, vocab_size=256)
    check("vocab_size 256 means zero merges", t3.merges == []
          and t3.vocab_size == 256)
    try:
        BPETokenizer().train(text, vocab_size=100)
        check("vocab_size < 256 rejected", False)
    except ValueError:
        check("vocab_size < 256 rejected", True)

    t4 = BPETokenizer(); t4.train("aaa", vocab_size=300)
    check("corpus exhausted before vocab reached", len(t4.merges) < 44)


def test_bpe_persistence():
    print("\n[5] BPE save/load + checkpoint state")
    from ai_core.tokenizer import (BPETokenizer, codec_from_checkpoint,
                                   tokenizer_state)

    tok = BPETokenizer()
    tok.train("persistent merges test " * 50, vocab_size=320)
    sample = "persistent merges test!"
    ids = tok.encode(sample)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tok.json")
        tok.save(path)
        loaded = BPETokenizer.load(path)
        check("file round-trip: same merges", loaded.merges == tok.merges)
        check("file round-trip: same encoding", loaded.encode(sample) == ids)
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        check("json format tag", raw.get("format") == "aira-bpe-1")

        state = tokenizer_state(tok)
        check("tokenizer_state is json-able",
              json.loads(json.dumps(state)) == state)
        check("tokenizer_state type", state["type"] == "bpe")
        from_state_tok = BPETokenizer.from_state(state)
        check("from_state round-trip", from_state_tok.encode(sample) == ids)

        ckpt = {"model": {}, "config": {}, "tokenizer": state}
        codec = codec_from_checkpoint(ckpt)
        check("codec_from_checkpoint returns it",
              isinstance(codec, BPETokenizer) and codec.vocab_size == tok.vocab_size)

    # Legacy char checkpoints (no tokenizer key)
    legacy = {"chars": sorted(set("abc"))}
    char_codec = codec_from_checkpoint(legacy)
    check("legacy checkpoint -> CharCodec", char_codec.type_name == "char"
          and char_codec.vocab_size == 3)
    # Explicit char checkpoint with new-style tokenizer state
    modern = {"chars": sorted(set("abc")), "tokenizer":
              {"type": "char", "vocab_size": 3}}
    check("modern char checkpoint still works",
          codec_from_checkpoint(modern).type_name == "char")
    # BPE checkpoints may omit 'chars' (vocab derives from merges: 256+n)
    bpe_codec = codec_from_checkpoint({
        "tokenizer": {"type": "bpe", "vocab_size": 257,
                      "merges": [[97, 97]]}})
    check("bpe checkpoint without chars key",
          isinstance(bpe_codec, BPETokenizer)
          and bpe_codec.vocab_size == 257
          and bpe_codec.decode(bpe_codec.encode("aa")) == "aa")


def test_bpe_beats_char_on_context():
    print("\n[6] Integration: BPE packs more text into one context window")
    import torch
    from ai_core.model import EmotionAwareGPT
    from ai_core.tokenizer import BPETokenizer, CharCodec

    torch.manual_seed(0)
    text = ("def aira_think():\n    return emotion + text\n"
            * 100)

    char_codec = CharCodec(sorted(set(text)))
    char_ids = char_codec.encode(text)

    tok = BPETokenizer()
    bpe_ids = tok.train(text, vocab_size=512)

    block = 64
    cfg = {"n_embd": 64, "n_head": 4, "n_layer": 2,
           "block_size": block, "dropout": 0.0}
    m_char = EmotionAwareGPT(char_codec.vocab_size, cfg)
    m_bpe = EmotionAwareGPT(tok.vocab_size, cfg)
    check("both models build", m_char is not None and m_bpe is not None)

    chars_in_block_char = min(block, len(char_ids))          # 1 char/token
    chars_in_block_bpe = len(tok.decode(
        bpe_ids[:block]))                                    # many chars/token
    check("bpe window holds more text",
          chars_in_block_bpe > chars_in_block_char,
          f"bpe {chars_in_block_bpe} vs char {chars_in_block_char}")

    # Checkpoints written the new way load through codec_from_checkpoint
    from ai_core.tokenizer import codec_from_checkpoint
    ckpt_char = {"chars": char_codec.chars,
                 "tokenizer": {"type": "char", "vocab_size":
                               char_codec.vocab_size}}
    ckpt_bpe = tok
    check("char codec from new-style ckpt",
          codec_from_checkpoint(ckpt_char).type_name == "char")
    check("bpe codec from trained tokenizer",
          codec_from_checkpoint({
              "tokenizer": {"type": "bpe",
                            "vocab_size": tok.vocab_size,
                            "merges": [list(p) for p in tok.merges]}}
          ).encode(text[:20]) == tok.encode(text[:20]))

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_bpe_path = os.path.join(tmp, "bpe.pt")
        torch.save({"model": m_bpe.state_dict(),
                    "config": cfg,
                    "tokenizer": {"type": "bpe", "vocab_size": tok.vocab_size,
                                  "merges": [list(p) for p in tok.merges]}},
                   ckpt_bpe_path)
        from ai_core.agent import load_model
        model, codec = load_model(ckpt_bpe_path, device="cpu")
        check("agent.load_model handles bpe checkpoints",
              codec.type_name == "bpe"
              and model.token_emb.num_embeddings == tok.vocab_size)


def main():
    test_char_codec()
    test_bpe_train_and_roundtrip()
    test_bpe_robustness()
    test_bpe_determinism_and_vocab()
    test_bpe_persistence()
    test_bpe_beats_char_on_context()
    print(f"\n{'=' * 50}")
    print(f"PASSED: {PASS}  FAILED: {FAIL}")
    return FAIL == 0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

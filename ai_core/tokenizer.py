"""Tokenizers for Aira's EmotionAwareGPT — pure Python, from scratch.

Two codecs live here and share one interface (``encode`` / ``decode`` /
``vocab_size``):

* ``CharCodec``    — the original one-character-per-token scheme.
* ``BPETokenizer`` — byte-level Byte Pair Encoding. Learn merges from the
  training text, then every input (any language, any emoji, any byte)
  tokenizes into 0..vocab_size-1. Subwords pack roughly 2-4x more text
  into the same context window than characters, which is Aira's cheapest
  path to a "bigger brain" without touching the transformer itself.

No external libraries, no pretrained data: merges are learned from your
own corpus at train time and travel inside the checkpoint.
"""
import json
import os


# ---------------------------------------------------------------- codecs --

class CharCodec:
    """One character = one token (the original Aira vocabulary)."""

    type_name = "char"

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """Chars to ids; unknown characters are dropped (legacy behavior)."""
        known = "".join(c for c in text if c in self.stoi)
        return [self.stoi[c] for c in known]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class BPETokenizer:
    """Byte-level Byte Pair Encoding, learned from Aira's own corpus.

    Every token id < 256 is a raw UTF-8 byte, ids >= 256 are learned
    merges. Because the base alphabet is all 256 bytes, *any* text
    encodes — no unknown characters, ever.

    Usage:
        tok = BPETokenizer()
        ids = tok.train(text, vocab_size=512)   # learn merges
        tok.save("tokenizer.json")

        tok2 = BPETokenizer.load("tokenizer.json")
        ids = tok2.encode("Hello!")              # list[int]
        text = tok2.decode(ids)                  # exact round-trip
    """

    type_name = "bpe"
    FORMAT = "aira-bpe-1"

    def __init__(self):
        self.merges = []            # list of (a, b) pairs, lowest rank first
        self.ranks = {}             # (a, b) -> rank (== merge order == new id - 256)
        self._vocab = None          # id -> bytes, built lazily

    # -- training ---------------------------------------------------------

    def train(self, text, vocab_size=512, verbose=False):
        """Learn merges until the vocabulary reaches ``vocab_size``.

        Returns the token id list for ``text`` (so trainers can reuse
        it without re-encoding). Byte-level: never fails on unicode.
        """
        if vocab_size < 256:
            raise ValueError("BPE vocab_size must be at least 256 (one id per byte)")
        ids = list(text.encode("utf-8"))
        num_merges = max(0, int(vocab_size) - 256)
        self.merges = []
        for i in range(num_merges):
            counts = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            best = max(counts, key=counts.get)   # deterministic tie-break: first seen
            new_id = 256 + i
            ids = self._merge(ids, best, new_id)
            self.merges.append(best)
            if verbose and (i + 1) % 32 == 0:
                print(f"  bpe merge {i + 1}/{num_merges}: {best} -> {new_id}")
        self._reindex()
        return ids

    @staticmethod
    def _merge(ids, pair, new_id):
        a, b = pair
        out = []
        i = 0
        n = len(ids)
        while i < n:
            if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # -- state ------------------------------------------------------------

    def _reindex(self):
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._vocab = None

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def build_vocab(self):
        """id -> bytes for every token (base bytes + learned merges)."""
        if self._vocab is None:
            vocab = [bytes([i]) for i in range(256)]
            for new_id, (a, b) in enumerate(self.merges, start=256):
                vocab.append(vocab[a] + vocab[b])
            self._vocab = vocab
        return self._vocab

    # -- encoding ----------------------------------------------------------

    def encode(self, text):
        """Text to ids. Greedy lowest-rank merges over UTF-8 bytes."""
        ids = list(str(text or "").encode("utf-8"))
        ranks = self.ranks
        while len(ids) >= 2:
            best = None
            best_rank = None
            for pair in zip(ids, ids[1:]):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = pair
            if best is None:
                break
            ids = self._merge(ids, best, 256 + best_rank)
        return ids

    def decode(self, ids):
        vocab = self.build_vocab()
        raw = b"".join(vocab[i] for i in ids if 0 <= i < len(vocab))
        return raw.decode("utf-8", errors="replace")

    # -- persistence ---------------------------------------------------------

    def save(self, path):
        payload = {
            "format": self.FORMAT,
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
        }
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != cls.FORMAT:
            raise ValueError(f"{path} is not an {cls.FORMAT} tokenizer file")
        tok = cls()
        tok.merges = [tuple(int(x) for x in pair)
                      for pair in payload.get("merges", [])]
        tok._reindex()
        return tok

    @classmethod
    def from_state(cls, state):
        """Build from the ``checkpoint['tokenizer']`` dict saved by train.py."""
        tok = cls()
        tok.merges = [tuple(int(x) for x in pair)
                      for pair in state.get("merges", [])]
        tok._reindex()
        return tok

    @classmethod
    def from_trained_merges(cls, merge_list):
        """Test seam: construct directly from an already-trained merge list."""
        tok = cls()
        tok.merges = [tuple(pair) for pair in merge_list]
        tok._reindex()
        return tok


# ------------------------------------------------------- checkpoint glue --

def codec_from_checkpoint(checkpoint):
    """Pick the right codec for a checkpoint saved by train.py.

    Old checkpoints (no ``tokenizer`` key) are char-level.
    """
    state = checkpoint.get("tokenizer")
    if state and state.get("type") == BPETokenizer.type_name:
        return BPETokenizer.from_state(state)
    return CharCodec(checkpoint.get("chars") or [])


def tokenizer_state(codec):
    """The codec-describing dict stored inside checkpoints."""
    if isinstance(codec, BPETokenizer):
        return {
            "type": BPETokenizer.type_name,
            "vocab_size": codec.vocab_size,
            "merges": [list(pair) for pair in codec.merges],
        }
    return {"type": CharCodec.type_name, "vocab_size": codec.vocab_size}

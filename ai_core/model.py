"""
Core AI Model - Transformer-based LLM
"""
import math
import re
import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """Causal multi-head attention with a single fused QKV projection."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )

    def _split_heads(self, tensor, batch, time):
        return tensor.view(
            batch, time, self.n_head, self.head_size
        ).transpose(1, 2)

    def forward(self, x):
        out, _ = self.forward_with_cache(x, None, masked=True)
        return out

    def forward_with_cache(self, x, cache, masked):
        """Attend over x, appending to a (key, value) cache when given.

        `masked` must be True whenever x holds more than one token, so the
        causal mask hides future positions. Single-token steps with a warm
        cache need no mask: the newest token always sees every cached key.
        """
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = self._split_heads(q, batch, time)
        k = self._split_heads(k, batch, time)
        v = self._split_heads(v, batch, time)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        if masked:
            scores = scores.masked_fill(
                self.mask[:time, :time] == 0, float("-inf")
            )
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(batch, time, channels)
        return self.dropout(self.proj(out)), (k, v)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))

    def forward_with_cache(self, x, cache, masked):
        h, cache = self.attn.forward_with_cache(self.ln1(x), cache, masked)
        x = x + h
        return x + self.ff(self.ln2(x)), cache


class EmotionAwareGPT(nn.Module):
    """GPT model with emotion awareness"""

    def __init__(self, vocab_size, cfg):
        super().__init__()
        n_embd = cfg["n_embd"]
        block_size = cfg["block_size"]
        dropout = cfg["dropout"]
        n_layer = cfg["n_layer"]

        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.emotion_emb = nn.Embedding(8, n_embd)  # 8 emotion categories

        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, cfg["n_head"], block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, emotion_id=None, targets=None):
        t = idx.size(1)
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        # Add emotion context if provided
        if emotion_id is not None:
            if not torch.is_tensor(emotion_id):
                emotion_id = torch.tensor([emotion_id], device=idx.device)
            x = x + self.emotion_emb(emotion_id)

        x = self.blocks(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    def _emotion_vector(self, emotion_id, device):
        if emotion_id is None:
            return None
        return self.emotion_emb(
            torch.tensor([emotion_id], device=device)
        )

    def _logits_with_caches(self, x, caches, masked):
        for i, block in enumerate(self.blocks):
            x, caches[i] = block.forward_with_cache(
                x, caches[i], masked
            )
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx, count, temperature=0.8, emotion_id=None):
        self.eval()
        idx = idx.clone()
        if 0 < idx.size(1) + count <= self.block_size:
            return self._generate_cached(idx, count, temperature, emotion_id)
        for _ in range(count):
            logits = self(idx[:, -self.block_size:], emotion_id)[:, -1, :]
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    @torch.no_grad()
    def _generate_cached(self, idx, count, temperature, emotion_id):
        """Incremental decoding with a per-layer KV cache.

        Exact for idx.size(1) + count <= block_size: positions never wrap,
        so a cached token keeps the same embedding it would get in a full
        forward. Longer sequences fall back to the recompute path, which
        crops the context window and shifts positions.
        """
        device = idx.device
        emotion_vector = self._emotion_vector(emotion_id, device)
        prompt_len = idx.size(1)
        caches = [None] * len(self.blocks)
        for step in range(count):
            if step == 0:
                tokens, positions, masked = idx, torch.arange(
                    prompt_len, device=device
                ), True
            else:
                tokens = idx[:, -1:]
                positions = torch.tensor(
                    [idx.size(1) - 1], device=device
                )
                masked = False
            x = self.token_emb(tokens) + self.pos_emb(positions)
            if emotion_vector is not None:
                x = x + emotion_vector
            logits = self._logits_with_caches(
                x, caches, masked
            )[:, -1, :]
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


_QKV_PART = {"query": 0, "key": 1, "value": 2}
_LEGACY_HEAD_KEY = re.compile(
    r"^(.*\.attn)\.heads\.(\d+)\.(key|query|value)\.weight$"
)
_LEGACY_HEAD_MASK = re.compile(r"^(.*\.attn)\.heads\.(\d+)\.mask$")


def fuse_legacy_state_dict(state_dict):
    """Remap per-head attention weights from older checkpoints.

    Legacy checkpoints store one key/query/value Linear per head
    (``blocks.N.attn.heads.H.<part>.weight``); the fused model keeps a
    single ``blocks.N.attn.qkv.weight`` laid out as [Q; K; V] with heads
    grouped inside each part. Checkpoints saved by this module pass
    through unchanged.
    """
    fused = {}
    parts_by_attn = {}
    for name, tensor in state_dict.items():
        match = _LEGACY_HEAD_KEY.match(name)
        if match is None:
            if _LEGACY_HEAD_MASK.match(name) is None:
                fused[name] = tensor
            continue
        attn, head, part = match.group(1), int(match.group(2)), match.group(3)
        parts_by_attn.setdefault(attn, []).append((head, part, tensor))
    for attn, parts in parts_by_attn.items():
        parts.sort()
        head_size, n_embd = parts[0][2].shape
        weight = torch.zeros(
            3 * n_embd, n_embd, dtype=parts[0][2].dtype
        )
        for head, part, tensor in parts:
            offset = _QKV_PART[part] * n_embd + head * head_size
            weight[offset:offset + head_size] = tensor
        fused[attn + ".qkv.weight"] = weight
    return fused

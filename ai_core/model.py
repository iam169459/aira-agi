"""
Core AI Model - Transformer-based LLM
"""
import math
import re
import torch
import torch.nn as nn
from torch.nn import functional as F


def sampling_probs(logits, recent, temperature=0.8, top_k=None, top_p=1.0,
                   repetition_penalty=1.0):
    """Turn raw next-token logits into a sampling distribution.

    Applies temperature, repetition penalty (over the token ids in
    `recent`), top-k truncation, and nucleus (top-p) filtering — in
    that order. With the defaults (top_k=None, top_p=1.0,
    repetition_penalty=1.0) this is exactly softmax(logits / temperature).
    """
    logits = logits / max(temperature, 1e-5)
    if repetition_penalty and repetition_penalty != 1.0 and recent:
        tokens = sorted(set(recent))
        index = torch.tensor(tokens, dtype=torch.long,
                             device=logits.device)
        index = index.view(1, -1).expand(logits.size(0), -1)
        vals = torch.gather(logits, -1, index)
        penalized = torch.where(vals > 0, vals / repetition_penalty,
                                vals * repetition_penalty)
        logits = logits.scatter(-1, index, penalized)
    if top_k is not None and 0 < top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True,
                                               dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits)
    return F.softmax(logits, dim=-1)


class MultiHeadAttention(nn.Module):
    """Causal multi-head attention with a single fused QKV projection.

    ``use_rope`` enables rotary position embeddings (the Llama/Mistral
    scheme): Q/K are rotated by an angle proportional to their absolute
    position, which removes the hard context cap that learned position
    embeddings impose. Legacy models (use_rope=False) are byte-identical
    to the previous implementation.
    """

    def __init__(self, n_embd, n_head, block_size, dropout,
                 use_rope=False):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.use_rope = use_rope
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )
        if use_rope:
            inv_freq = 1.0 / (10000.0 ** (
                torch.arange(0, self.head_size, 2, dtype=torch.float32)
                / self.head_size))
            self.register_buffer("inv_freq", inv_freq,
                                 persistent=False)   # derived, not learned

    def _split_heads(self, tensor, batch, time):
        return tensor.view(
            batch, time, self.n_head, self.head_size
        ).transpose(1, 2)

    def _rotary_tables(self, positions, dtype, device):
        angles = positions[:, None].float() * self.inv_freq[None, :]
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def forward(self, x):
        out, _ = self.forward_with_cache(x, None, masked=True)
        return out

    def forward_with_cache(self, x, cache, masked, rope_offset=0):
        """Attend over x, appending to a (key, value) cache when given.

        `masked` must be True whenever x holds more than one token, so the
        causal mask hides future positions. Single-token steps with a warm
        cache need no mask: the newest token always sees every cached key.
        With RoPE, ``rope_offset`` is the absolute position of x[:, 0]
        (only nonzero for the sliding-window recompute path); per-step K/V
        positions are derived from the cache length automatically.
        """
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = self._split_heads(q, batch, time)
        k = self._split_heads(k, batch, time)
        v = self._split_heads(v, batch, time)
        if self.use_rope:
            start = (cache[0].size(2) + rope_offset
                     if cache is not None else rope_offset)
            positions = torch.arange(start, start + time, device=x.device)
            cos, sin = self._rotary_tables(positions, q.dtype, x.device)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        if masked:
            if time <= self.mask.size(0):
                causal = self.mask
            else:   # RoPE full forwards may exceed the training window
                causal = torch.tril(
                    torch.ones(time, time, device=x.device))
            scores = scores.masked_fill(
                causal[:time, :time] == 0, float("-inf")
            )
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(batch, time, channels)
        return self.dropout(self.proj(out)), (k, v)


def _apply_rope(x, cos, sin):
    """Rotate consecutive (even, odd) feature pairs of x by their angle.

    x: (batch, heads, time, head_size); cos/sin: (time, head_size // 2)
    (or pre-broadcast to (1, 1, time, head_size // 2)).
    Pure rotation — preserves vector norms, mixes position into attention.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    if cos.dim() == 2:              # broadcast over batch and heads
        cos = cos[None, None]
        sin = sin[None, None]
    return torch.stack(
        (x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
    ).flatten(-2)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward"""

    def __init__(self, n_embd, n_head, block_size, dropout,
                 use_rope=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout,
                                       use_rope=use_rope)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope_offset=0):
        h, _ = self.attn.forward_with_cache(
            self.ln1(x), None, True, rope_offset)
        x = x + h
        return x + self.ff(self.ln2(x))

    def forward_with_cache(self, x, cache, masked, rope_offset=0):
        h, cache = self.attn.forward_with_cache(
            self.ln1(x), cache, masked, rope_offset)
        x = x + h
        return x + self.ff(self.ln2(x)), cache


class EmotionAwareGPT(nn.Module):
    """GPT model with emotion awareness.

    Set ``cfg["rope"] = True`` for rotary position embeddings (no learned
    pos_emb, generation not capped at block_size). Default cfg keeps the
    legacy learned embeddings so old checkpoints load unchanged.
    """

    def __init__(self, vocab_size, cfg):
        super().__init__()
        n_embd = cfg["n_embd"]
        block_size = cfg["block_size"]
        dropout = cfg["dropout"]
        n_layer = cfg["n_layer"]
        self.use_rope = bool(cfg.get("rope", False))

        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = (None if self.use_rope
                        else nn.Embedding(block_size, n_embd))
        self.emotion_emb = nn.Embedding(8, n_embd)  # 8 emotion categories

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, cfg["n_head"], block_size, dropout,
                             use_rope=self.use_rope)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, emotion_id=None, targets=None, rope_offset=0):
        t = idx.size(1)
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(t, device=idx.device)
            x = x + self.pos_emb(pos)

        # Add emotion context if provided
        if emotion_id is not None:
            if not torch.is_tensor(emotion_id):
                emotion_id = torch.tensor([emotion_id], device=idx.device)
            x = x + self.emotion_emb(emotion_id)

        for block in self.blocks:
            x = block(x, rope_offset=rope_offset)
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

    def _logits_with_caches(self, x, caches, masked, rope_offset=0):
        for i, block in enumerate(self.blocks):
            x, caches[i] = block.forward_with_cache(
                x, caches[i], masked, rope_offset
            )
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx, count, temperature=0.8, emotion_id=None,
                 top_k=None, top_p=1.0, repetition_penalty=1.0):
        self.eval()
        idx = idx.clone()
        fits = 0 < idx.size(1) + count <= self.block_size
        if fits or (self.use_rope and 0 < idx.size(1) <= self.block_size):
            return self._generate_cached(idx, count, temperature,
                                         emotion_id, top_k, top_p,
                                         repetition_penalty)
        for _ in range(count):
            ctx = idx[:, -self.block_size:]
            offset = idx.size(1) - ctx.size(1)   # absolute pos of ctx[0]
            logits = self(ctx, emotion_id,
                          rope_offset=offset)[:, -1, :]
            probs = sampling_probs(logits, idx[0].tolist(), temperature,
                                   top_k, top_p, repetition_penalty)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    @torch.no_grad()
    def _generate_cached(self, idx, count, temperature, emotion_id,
                         top_k=None, top_p=1.0, repetition_penalty=1.0):
        """Incremental decoding with a per-layer KV cache.

        With learned embeddings this is exact only while
        idx.size(1) + count <= block_size (positions never wrap). With
        RoPE there is no learned-position table, so single-token steps
        can run arbitrarily far past block_size: the KV cache keeps keys
        at their absolute angles and the causal pattern stays correct.
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
            x = self.token_emb(tokens)
            if self.pos_emb is not None:
                x = x + self.pos_emb(positions)
            if emotion_vector is not None:
                x = x + emotion_vector
            logits = self._logits_with_caches(
                x, caches, masked
            )[:, -1, :]
            probs = sampling_probs(logits, idx[0].tolist(), temperature,
                                   top_k, top_p, repetition_penalty)
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

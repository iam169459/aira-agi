"""
Core AI Model - Transformer-based LLM
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    """Single attention head with causal masking"""
    
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        t = x.size(1)
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        scores = scores.masked_fill(
            self.mask[:t, :t] == 0, float("-inf")
        )
        return F.softmax(scores, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel"""
    
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            AttentionHead(n_embd, head_size, block_size)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))


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

    def forward(self, idx, emotion_id=None):
        t = idx.size(1)
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Add emotion context if provided
        if emotion_id is not None:
            x = x + self.emotion_emb(emotion_id)
        
        x = self.blocks(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx, count, temperature=0.8, emotion_id=None):
        self.eval()
        for _ in range(count):
            logits = self(idx[:, -self.block_size:], emotion_id)[:, -1, :]
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
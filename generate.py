import argparse
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
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


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        n_embd = cfg["n_embd"]
        block_size = cfg["block_size"]
        dropout = cfg["dropout"]
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embd, cfg["n_head"], block_size, dropout
            )
            for _ in range(cfg["n_layer"])
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        t = idx.size(1)
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        return self.head(self.ln(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, count, temperature):
        self.eval()
        for _ in range(count):
            logits = self(idx[:, -self.block_size:])[:, -1, :]
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="tiny_model.pt")
    parser.add_argument("--prompt", default="A language model")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Checkpoints are created by this project and should be treated as trusted local files.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    chars = checkpoint["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        raise SystemExit(
            "Prompt contains characters not in training vocabulary: "
            + repr("".join(unknown))
            + ". Use a prompt with characters found in data.txt."
        )

    model = TinyGPT(len(chars), checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)
    output = model.generate(idx, args.tokens, args.temperature)[0].tolist()
    print("".join(itos[i] for i in output))


if __name__ == "__main__":
    main()

import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)

DATA_FILE = "data.txt"
CHECKPOINT = "tiny_model.pt"

# Conservative starter settings for a 16 GB Mac.
BLOCK_SIZE = 64
BATCH_SIZE = 8
MAX_STEPS = 500
EVAL_EVERY = 100
LEARNING_RATE = 3e-4

N_EMBD = 96
N_HEAD = 4
N_LAYER = 3
DROPOUT = 0.1

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("Create data.txt before training.")

with open(DATA_FILE, encoding="utf-8") as f:
    text = f.read()

if len(text) < 2 * BLOCK_SIZE + 20:
    raise ValueError(
        f"data.txt needs at least {2 * BLOCK_SIZE + 20} characters "
        "for this starter configuration. Add more text."
    )

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

split = int(len(encoded) * 0.9)
train_data, val_data = encoded[:split], encoded[split:]
if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
    raise ValueError("Add more text so both splits exceed BLOCK_SIZE.")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)
print("Vocabulary size:", len(chars))
print("Characters:", len(text))


def get_batch(source):
    starts = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in starts])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in starts])
    return x.to(device), y.to(device)


class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        _, t, _ = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        scores = scores.masked_fill(
            self.mask[:t, :t] == 0, float("-inf")
        )
        weights = F.softmax(scores, dim=-1)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.heads = nn.ModuleList(
            [AttentionHead(head_size) for _ in range(N_HEAD)]
        )
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.attn = MultiHeadAttention()
        self.ff = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(N_LAYER)]
        )
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        positions = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        logits = self.head(self.ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8):
        self.eval()
        temperature = max(float(temperature), 1e-5)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


model = TinyGPT(len(chars)).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}
    for name, source in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(5):
            x, y = get_batch(source)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[name] = round(sum(losses) / len(losses), 4)
    model.train()
    return results


model.train()
for step in range(MAX_STEPS):
    if step % EVAL_EVERY == 0:
        print(f"step {step}: {estimate_loss()}")
    x, y = get_batch(train_data)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print("Final loss:", estimate_loss())
torch.save(
    {
        "model": model.cpu().state_dict(),
        "chars": chars,
        "config": {
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "dropout": DROPOUT,
        },
    },
    CHECKPOINT,
)
print(f"Saved checkpoint to {CHECKPOINT}")

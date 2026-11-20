import os

import torch
from torch.nn import functional as F

from ai_core.model import EmotionAwareGPT

torch.manual_seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT, "data", "code_corpus.txt")
CHECKPOINT = os.path.join(ROOT, "tiny_model.pt")


# Conservative starter settings for a 16 GB Mac.
BLOCK_SIZE = 2048
BATCH_SIZE = 1
MAX_STEPS = 1000
EVAL_EVERY = 200
LEARNING_RATE = 3e-4

N_EMBD = 1440
N_HEAD = 16
N_LAYER = 16
DROPOUT = 0.1

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("Create data/code_corpus.txt before training.")

with open(DATA_FILE, encoding="utf-8") as f:
    text = f.read()

if len(text) < 2 * BLOCK_SIZE + 20:
    raise ValueError(
        f"{DATA_FILE} needs at least {2 * BLOCK_SIZE + 20} characters "
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


config = {
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "n_head": N_HEAD,
    "n_layer": N_LAYER,
    "dropout": DROPOUT,
}
model = EmotionAwareGPT(len(chars), config).to(device)
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
            _, loss = model(x, targets=y)
            losses.append(loss.item())
        results[name] = round(sum(losses) / len(losses), 4)
    model.train()
    return results


model.train()
for step in range(MAX_STEPS):
    if step % EVAL_EVERY == 0:
        print(f"step {step}: {estimate_loss()}")
    x, y = get_batch(train_data)
    _, loss = model(x, targets=y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print("Final loss:", estimate_loss())
torch.save(
    {
        "model": model.cpu().state_dict(),
        "chars": chars,
        "config": config,
    },
    CHECKPOINT,
)
print(f"Saved checkpoint to {CHECKPOINT}")

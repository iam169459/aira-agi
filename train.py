import argparse
import os

import torch
from torch.nn import functional as F

from ai_core.model import EmotionAwareGPT

torch.manual_seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT, "data", "code_corpus.txt")
CHECKPOINT = os.path.join(ROOT, "tiny_model.pt")

# Training profiles. "tiny" is the original starter; "beast" is several
# sizes up on a combined conversation + code corpus.
PRESETS = {
    "tiny": {
        "data": "code_corpus.txt", "block_size": 64,
        "batch_size": 8, "max_steps": 500, "eval_every": 100,
        "lr": 3e-4, "n_embd": 96, "n_head": 4, "n_layer": 3,
    },
    "base": {
        "data": "daily_conversations.txt", "block_size": 128,
        "batch_size": 8, "max_steps": 1500, "eval_every": 300,
        "lr": 3e-4, "n_embd": 192, "n_head": 6, "n_layer": 5,
    },
    "beast": {
        "data": "beast_corpus.txt", "block_size": 256,
        "batch_size": 8, "max_steps": 4000, "eval_every": 500,
        "lr": 3e-4, "n_embd": 320, "n_head": 8, "n_layer": 8,
    },
    "titan": {
        "data": "beast_corpus.txt", "block_size": 512,
        "batch_size": 8, "max_steps": 8000, "eval_every": 500,
        "lr": 3e-4, "n_embd": 512, "n_head": 8, "n_layer": 12,
    },
}
DROPOUT = 0.1


def build_corpus(root=ROOT):
    """Concatenate conversations, code, and general data into the
    beast training corpus (data/beast_corpus.txt)."""
    parts = []
    for name in ("daily_conversations.txt",
                 "combined_training_data.txt",
                 "code_corpus.txt"):
        path = os.path.join(root, "data", name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                parts.append(fh.read().strip() + "\n\n")
    if not parts:
        raise FileNotFoundError(
            f"no training data found in {os.path.join(root, 'data')}")
    out = os.path.join(root, "data", "beast_corpus.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="train the EmotionAwareGPT checkpoint (tiny_model.pt)")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS),
                        help="training profile (default: tiny)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override the preset's step count")
    parser.add_argument("--data", default=None,
                        help="override the preset's training file")
    parser.add_argument("--out", default=None,
                        help="checkpoint output path (default: tiny_model.pt)")
    args = parser.parse_args()
    train(args.preset, steps=args.steps, data_file=args.data,
          out_path=args.out)


def train(preset="tiny", steps=None, data_file=None, out_path=None,
          root=ROOT):
    cfg = dict(PRESETS[preset])
    data_file = data_file or cfg["data"]
    if data_file == "beast_corpus.txt" and not os.path.exists(
            os.path.join(root, "data", data_file)):
        build_corpus(root)
    DATA_FILE = os.path.join(root, "data", data_file)
    CHECKPOINT = out_path or os.path.join(root, "tiny_model.pt")
    steps = steps or cfg["max_steps"]
    BLOCK_SIZE = cfg["block_size"]
    BATCH_SIZE = cfg["batch_size"]
    EVAL_EVERY = cfg["eval_every"]
    LEARNING_RATE = cfg["lr"]
    N_EMBD = cfg["n_embd"]
    N_HEAD = cfg["n_head"]
    N_LAYER = cfg["n_layer"]

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
    for step in range(steps):
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


if __name__ == "__main__":
    main()

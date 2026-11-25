import argparse
import os

import torch
from torch.nn import functional as F

from ai_core.model import EmotionAwareGPT, fuse_legacy_state_dict

torch.manual_seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT, "data", "code_corpus.txt")
CHECKPOINT = os.path.join(ROOT, "tiny_model.pt")

# Training profiles. "tiny" is the original starter; "beast" is several
# sizes up. "xl", "xxl" and "titan" scale the same architecture up to
# Kaggle-GPU sizes and all train on the combined beast corpus.
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
    "xl": {
        "data": "beast_corpus.txt", "block_size": 384,
        "batch_size": 16, "grad_accum": 1, "max_steps": 10000,
        "eval_every": 500,
        "lr": 3e-4, "n_embd": 512, "n_head": 8, "n_layer": 10,
    },
    "xxl": {
        "data": "beast_corpus.txt", "block_size": 512,
        "batch_size": 16, "grad_accum": 2, "max_steps": 20000,
        "eval_every": 1000,
        "lr": 2.5e-4, "n_embd": 768, "n_head": 12, "n_layer": 12,
    },
    "titan": {
        "data": "beast_corpus.txt", "block_size": 512,
        "batch_size": 8, "grad_accum": 4, "max_steps": 30000,
        "eval_every": 1000,
        "lr": 2.5e-4, "n_embd": 1024, "n_head": 16, "n_layer": 16,
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


# One canonical model file: training resumes it and it keeps growing.
DEFAULT_CHECKPOINT_NAME = "aira_model.pt"


def main():
    parser = argparse.ArgumentParser(
        description="train EmotionAwareGPT into one growing checkpoint "
                    "(aira_model.pt); re-running resumes that same model")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS),
                        help="training profile (default: tiny)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override the preset's step count")
    parser.add_argument("--data", default=None,
                        help="override the preset's training file")
    parser.add_argument("--out", default=None,
                        help=f"checkpoint path (default: {DEFAULT_CHECKPOINT_NAME})")
    parser.add_argument("--fresh", action="store_true",
                        help="discard the existing checkpoint and retrain "
                             "from scratch (needed when deliberately "
                             "switching to a different preset size)")
    args = parser.parse_args()
    train(args.preset, steps=args.steps, data_file=args.data,
          out_path=args.out, fresh=args.fresh)


def train(preset="tiny", steps=None, data_file=None, out_path=None,
          root=ROOT, fresh=False):
    """Train EmotionAwareGPT into one growing checkpoint.

    ``steps`` counts optimizer steps in THIS run (each consumes
    ``grad_accum`` micro-batches). If the existing checkpoint matches this
    preset's architecture, training resumes from its last step and saves
    periodically. Pass ``fresh=True`` to retrain from scratch.
    """
    cfg = dict(PRESETS[preset])
    data_file = data_file or cfg["data"]
    if data_file == "beast_corpus.txt" and not os.path.exists(
            os.path.join(root, "data", data_file)):
        build_corpus(root)
    DATA_FILE = os.path.join(root, "data", data_file)
    CHECKPOINT = out_path or os.path.join(root, DEFAULT_CHECKPOINT_NAME)
    steps = steps or cfg["max_steps"]
    BLOCK_SIZE = cfg["block_size"]
    BATCH_SIZE = cfg["batch_size"]
    GRAD_ACCUM = cfg.get("grad_accum", 1)
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

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device == "cuda"
    ndev = torch.cuda.device_count() if use_amp else 0
    if use_amp:
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_dtype == torch.float16)
    else:
        amp_dtype = torch.float32
        scaler = None
    if ndev > 1:
        BATCH_SIZE = BATCH_SIZE * ndev
    print("Device:", device,
          f"| GPUs: {ndev} (DataParallel)" if ndev > 1 else "",
          "| AMP:", amp_dtype if use_amp else "off")
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
    resume_state = None
    if not fresh and os.path.exists(CHECKPOINT):
        try:
            saved = torch.load(CHECKPOINT, map_location="cpu",
                               weights_only=False)
        except Exception as exc:
            print(f"Ignoring unreadable {CHECKPOINT}: {exc}")
        else:
            saved_cfg = saved.get("config", {})
            same = (saved_cfg.get("n_embd") == N_EMBD
                    and saved_cfg.get("n_head") == N_HEAD
                    and saved_cfg.get("n_layer") == N_LAYER
                    and saved_cfg.get("block_size") == BLOCK_SIZE)
            if not same:
                raise ValueError(
                    f"{CHECKPOINT} holds a different model size "
                    f"({saved_cfg.get('n_embd')}x{saved_cfg.get('n_layer')} "
                    f"vs this preset {N_EMBD}x{N_LAYER}). Use --fresh to "
                    "deliberately replace it, or --out to keep both.")
            else:
                resume_state = saved
                chars = saved["chars"]
                stoi = {c: i for i, c in enumerate(chars)}
                itos = {i: c for i, c in stoi.items()}
                known = "".join(c for c in text if c in stoi)
                encoded = torch.tensor([stoi[c] for c in known],
                                       dtype=torch.long)
                split = int(len(encoded) * 0.9)
                train_data, val_data = encoded[:split], encoded[split:]
                print(f"Resuming {CHECKPOINT} at step {saved.get('step', 0)}")
    model = EmotionAwareGPT(len(chars), config).to(device)
    if resume_state is not None:
        try:
            model.load_state_dict(resume_state["model"])
        except RuntimeError:
            model.load_state_dict(
                fuse_legacy_state_dict(resume_state["model"]))
    if use_amp:
        _fp_forward = model.forward
        def amp_forward(*fwd_args, **fwd_kwargs):
            with torch.autocast("cuda", dtype=amp_dtype):
                return _fp_forward(*fwd_args, **fwd_kwargs)
        model.forward = amp_forward
    if ndev > 1:
        model = torch.nn.DataParallel(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    net = model.module if isinstance(model, torch.nn.DataParallel) else model
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    if resume_state is not None and resume_state.get("optimizer"):
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_RATE
        except (ValueError, KeyError):
            print("Saved optimizer state is incompatible; using a fresh one.")


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


    def save_checkpoint(current_step):
        state = {key: value.detach().cpu()
                 for key, value in net.state_dict().items()}
        torch.save(
            {
                "model": state,
                "chars": chars,
                "config": config,
                "step": current_step,
                "optimizer": optimizer.state_dict(),
            },
            CHECKPOINT,
        )
        print(f"Saved checkpoint to {CHECKPOINT} at step {current_step}")

    model.train()
    step = resume_state.get("step", 0) if resume_state is not None else 0
    for micro in range(steps * GRAD_ACCUM):
        x, y = get_batch(train_data)
        _, loss = model(x, targets=y)
        loss = loss / GRAD_ACCUM
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (micro + 1) % GRAD_ACCUM:
            continue
        if step % EVAL_EVERY == 0:
            print(f"step {step}: {estimate_loss()}")
            save_checkpoint(step)
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

    print("Final loss:", estimate_loss())
    save_checkpoint(step)


if __name__ == "__main__":
    main()

import argparse
import os

import torch
from torch.nn import functional as F

from ai_core.model import EmotionAwareGPT, fuse_legacy_state_dict
from ai_core.tokenizer import (BPETokenizer, CharCodec, tokenizer_state)

torch.manual_seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT, "data", "code_corpus.txt")
CHECKPOINT = os.path.join(ROOT, "tiny_model.pt")

# Training profiles. "tiny" is the original starter; "beast" is several
# sizes up. "xl", "xxl" and "titan" scale the same architecture up to
# Kaggle-GPU sizes and all train on the combined beast corpus. "nova" is
# the modern-flagship: byte-level BPE + RoPE (see ai_core/tokenizer.py).
# A preset may specify "tokenizer"/"vocab_size"/"rope"; CLI flags
# (--tokenizer/--vocab-size/--rope) override those choices.
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
    # Modern-flagship preset: byte-level BPE (4096 vocab) + RoPE, so the
    # 512-token training window can still generate far beyond itself.
    "nova": {
        "data": "beast_corpus.txt", "block_size": 512,
        "batch_size": 8, "grad_accum": 2, "max_steps": 20000,
        "eval_every": 500,
        "lr": 3e-4, "n_embd": 768, "n_head": 12, "n_layer": 12,
        "tokenizer": "bpe", "vocab_size": 4096, "rope": True,
    },
}
DROPOUT = 0.1


def _dp_scalar(loss):
    """DataParallel gathers each replica's SCALAR loss into a 1-D
    vector (2 entries on 2 GPUs) - backward() then refuses a non-scalar
    ("grad can be implicitly created only for scalar outputs").
    Replicas split the batch evenly, so averaging the per-replica means
    returns the full-batch mean. Scalars and None pass through."""
    if loss is not None and loss.dim() != 0:
        loss = loss.mean()
    return loss


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
    parser.add_argument("--tokenizer", default=None,
                        choices=("char", "bpe"),
                        help="char = one character per token; bpe = "
                             "byte-level Byte Pair Encoding learned from "
                             "the corpus (packs 2-4x more text into the "
                             "same context window). Default: the preset's "
                             "choice (nova defaults to bpe, others char)")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="BPE vocabulary size (merges = vocab-size - "
                             "256); default: the preset's choice (512, or "
                             "4096 for nova)")
    parser.add_argument("--rope", action="store_true",
                        help="rotary position embeddings (Llama/Mistral "
                             "style): generation is no longer capped at "
                             "block_size (nova enables this by default)")
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping: stop after this many evals "
                             "without a val-loss improvement and keep the "
                             "best checkpoint (0 = train all steps; "
                             "default: 10)")
    args = parser.parse_args()
    train(args.preset, steps=args.steps, data_file=args.data,
          out_path=args.out, fresh=args.fresh,
          tokenizer=args.tokenizer, vocab_size=args.vocab_size,
          rope=args.rope, patience=args.patience)


def train(preset="tiny", steps=None, data_file=None, out_path=None,
          patience=None, eval_every=None,
          root=ROOT, fresh=False, tokenizer=None, vocab_size=None,
          rope=None):
    """Train EmotionAwareGPT into one growing checkpoint.

    ``steps`` counts optimizer steps in THIS run (each consumes
    ``grad_accum`` micro-batches). If the existing checkpoint matches this
    preset's architecture, tokenizer AND positional scheme, training
    resumes from its last step and saves periodically. Pass
    ``fresh=True`` to retrain from scratch. ``tokenizer="bpe"`` learns
    byte-level BPE merges from the corpus; ``rope=True`` swaps learned
    position embeddings for rotary embeddings (generation not capped at
    block_size). Unset arguments fall back to the preset's choices.
    """
    cfg = dict(PRESETS[preset])
    tokenizer = tokenizer or cfg.get("tokenizer", "char")
    vocab_size = vocab_size or cfg.get("vocab_size", 512)
    rope = bool(cfg.get("rope", False) if rope is None else rope)
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
    EVAL_EVERY = eval_every or cfg["eval_every"]
    # Early stopping: when the validation loss has not improved for this
    # many evals, stop and keep the best checkpoint. 0 disables it.
    PATIENCE = patience if patience is not None else 10
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

    use_bpe = tokenizer == "bpe"
    tok = None
    if use_bpe:
        if vocab_size < 512:
            print("Note: tiny BPE vocabularies (<512) learn only a few "
                  "merges; 512+ recommended.")
        tok = BPETokenizer()
        encoded = torch.tensor(
            tok.train(text, vocab_size=vocab_size), dtype=torch.long)
        vocab = tok.vocab_size
        compressed = len(text) / max(1, len(encoded))
        print(f"Tokenizer: byte-level BPE | merges: {len(tok.merges)} "
              f"| {compressed:.2f} chars per token")
    else:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        vocab = len(chars)
        print("Tokenizer: char (one character per token)")

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
          "| AMP:", amp_dtype if use_amp else "off",
          "| RoPE:", "on" if rope else "off")
    print("Vocabulary size:", vocab)
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
        "rope": rope,
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
            saved_tok = saved.get("tokenizer") or {}
            saved_kind = saved_tok.get("type", "char")
            same = (saved_cfg.get("n_embd") == N_EMBD
                    and saved_cfg.get("n_head") == N_HEAD
                    and saved_cfg.get("n_layer") == N_LAYER
                    and saved_cfg.get("block_size") == BLOCK_SIZE
                    and saved_kind == tokenizer
                    and bool(saved_cfg.get("rope", False)) == rope)
            if not same:
                raise ValueError(
                    f"{CHECKPOINT} holds a different model size, tokenizer "
                    f"or positional scheme ({saved_cfg.get('n_embd')}x"
                    f"{saved_cfg.get('n_layer')}, "
                    f"{saved_kind} tokenizer, "
                    f"rope={bool(saved_cfg.get('rope', False))} vs this "
                    f"preset {N_EMBD}x{N_LAYER}, {tokenizer}, "
                    f"rope={rope}). Use --fresh to "
                    "deliberately replace it, or --out to keep both.")
            else:
                resume_state = saved
                if use_bpe:
                    tok = BPETokenizer.from_state(saved["tokenizer"])
                    encoded = torch.tensor(tok.encode(text),
                                           dtype=torch.long)
                else:
                    chars = saved["chars"]
                    stoi = {c: i for i, c in enumerate(chars)}
                    itos = {i: c for i, c in stoi.items()}
                    known = "".join(c for c in text if c in stoi)
                    encoded = torch.tensor([stoi[c] for c in known],
                                           dtype=torch.long)
                    vocab = len(chars)
                split = int(len(encoded) * 0.9)
                train_data, val_data = encoded[:split], encoded[split:]
                print(f"Resuming {CHECKPOINT} at step {saved.get('step', 0)}")
    model = EmotionAwareGPT(vocab, config).to(device)
    if resume_state is not None:
        try:
            model.load_state_dict(resume_state["model"])
        except RuntimeError:
            model.load_state_dict(
                fuse_legacy_state_dict(resume_state["model"]))
    if use_amp:
        # autocast lives INSIDE the model (see model.amp_dtype): an
        # attribute is copied into every DataParallel replica, whereas an
        # externally replaced forward() would keep calling the original
        # cuda:0 module from inside every replica.
        model.amp_dtype = amp_dtype
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
                losses.append(_dp_scalar(loss).item())
            results[name] = round(sum(losses) / len(losses), 4)
        model.train()
        return results


    def save_checkpoint(current_step):
        state = {key: value.detach().cpu()
                 for key, value in net.state_dict().items()}
        if use_bpe:
            blob = {"tokenizer": tokenizer_state(tok)}
        else:
            blob = {"chars": chars, "tokenizer": tokenizer_state(
                CharCodec(chars))}
        torch.save(
            {
                "model": state,
                "config": config,
                "step": current_step,
                "optimizer": optimizer.state_dict(),
                "best_val": best_val if best_val != float("inf") else None,
                **blob,
            },
            CHECKPOINT,
        )
        print(f"Saved checkpoint to {CHECKPOINT} at step {current_step}")

    model.train()
    step = resume_state.get("step", 0) if resume_state is not None else 0
    best_val = float("inf")
    if resume_state is not None:
        best_val = resume_state.get("best_val", best_val)
    patience_left = PATIENCE
    stopped_early = False
    for micro in range(steps * GRAD_ACCUM):
        x, y = get_batch(train_data)
        _, loss = model(x, targets=y)
        loss = _dp_scalar(loss) / GRAD_ACCUM
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (micro + 1) % GRAD_ACCUM:
            continue
        if step % EVAL_EVERY == 0:
            metrics = estimate_loss()
            print(f"step {step}: {metrics}")
            if PATIENCE <= 0:
                save_checkpoint(step)
            elif metrics["val"] < best_val - 1e-4:
                # genuine improvement - keep this checkpoint
                best_val = metrics["val"]
                patience_left = PATIENCE
                save_checkpoint(step)
            else:
                patience_left -= 1
                print(f"  val not improving (best {best_val:.4f}) - "
                      f"early stop in {patience_left} evals")
                if patience_left <= 0:
                    stopped_early = True
                    break
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

    if stopped_early:
        # the file already holds the best-val checkpoint; sync the live
        # model to it so the session ends consistent with what's saved
        best_state = torch.load(CHECKPOINT, map_location=device,
                                weights_only=False)
        net.load_state_dict(best_state["model"])
        step = best_state["step"]
        print(f"Early-stopped: kept the best checkpoint "
              f"(step {step}, val {best_val:.4f}).")
        print("TIP: val loss rising while train loss falls means Aira has "
              "memorized this corpus - grow data/ (see "
              "docs/KAGGLE_TRAINING.md) before more steps help.")
    print("Final loss:", estimate_loss())
    if not stopped_early:
        save_checkpoint(step)


if __name__ == "__main__":
    main()

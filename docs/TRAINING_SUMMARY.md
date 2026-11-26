# Training Summary: Growing the One Aira Model

Aira is trained **from scratch** — no external AI APIs, no pretrained models. Every parameter,
including the tokenizer merges, is learned from our own corpora.

There is **ONE model**: `aira_model.pt`. Every training run resumes from it if the architecture
matches, so you never juggle `coding_model.pt` / `daily_model.pt` / `voice_model.pt` — you just
grow the single model with different data and bigger presets.

---

## The Preset Ladder (all in `train.py`)

| Preset | Params | block | n_embd | heads | layers | steps | Tokenizer | RoPE | Data (default) |
|--------|--------|-------|--------|-------|--------|-------|-----------|------|----------------|
| `tiny`   | 361,446    | 64  | 96  | 4  | 3  | 500    | char | no  | `code_corpus.txt` |
| `base`   | 2,287,206  | 128 | 192 | 6  | 5  | 1,500  | char | no  | `daily_conversations.txt` |
| `beast`  | 10,006,502 | 256 | 320 | 8  | 8  | 4,000  | char | no  | `beast_corpus.txt` |
| `xl`     | 31,814,758 | 384 | 384 | 8  | 10 | 10,000 | char | no  | `beast_corpus.txt` |
| `xxl`    | 85,584,486 | 512 | 512 | 12 | 12 | 20,000 | char | no  | `beast_corpus.txt` |
| `nova`   | 91,330,048 | 512 | 768 | 12 | 12 | 20,000 | **BPE 4096** | **yes** | `beast_corpus.txt` |
| `titan`  | 202,233,958 | 512 | 1024 | 16 | 16 | 30,000 | char | no | `beast_corpus.txt` |

- **BPE tokenizer** (`ai_core/tokenizer.py`): byte-level, built from scratch. Nova ships with
  vocab 4096 (~6.2 chars/token measured on the beast corpus — vs 1.0 for the char tokenizer).
  Merges live inside the checkpoint, so `aira_model.pt` is fully self-contained.
- **RoPE** (rotary position embeddings): opt-in, on for `nova`. Enables KV-cached generation that
  can run **past** the training block size, plus sliding-window cropping for long outputs.
- **AMP**: automatic on CUDA (`torch.bfloat16` when supported, fp16 otherwise) — applied via the
  replica-safe `model.amp_dtype` attribute, so it works correctly under `DataParallel` on 2×T4.

---

## What Each `data/` File Is For

| File | Size | Content |
|------|------|---------|
| `code_corpus.txt` | 391KB | Python/code examples and technical text |
| `daily_conversations.txt` | 17KB | Everyday conversation pairs |
| `combined_training_data.txt` | 14KB | Mixed code + conversation |
| `beast_corpus.txt` | 423KB | The merged large corpus (built by `train.build_corpus()` from daily + combined + code) |

Rebuild the beast corpus after editing the smaller ones:

```bash
python -c "import train; train.build_corpus()"
```

---

## `train_configs.py` — the Convenience Layer

`train_configs.py` is a friendly wrapper over the modern trainer. It **does not** duplicate any
model code — its generated scripts just `from train import train`.

| Config name | Maps to preset | Data | Best for |
|-------------|----------------|------|----------|
| `coding`   | `base`  | `code_corpus.txt` | Code-flavored training pass |
| `daily`    | `base`  | `daily_conversations.txt` | Conversation pass |
| `voice`    | `nova`  | beast corpus | Flagship quality for voice chats |
| `large`    | `nova`  | beast corpus | Same as voice — the GPU tier |
| `balanced`, `medium` | `beast` | beast corpus | The 10M CPU-trainable brain |
| `small`    | `base`  | daily corpus | Quick CPU runs |
| `tiny`     | `tiny`  | code corpus | Seconds-long smoke test |

Any `train.py` preset name (`tiny/base/beast/xl/xxl/nova/titan`) also works via
`get_config()` fall-through.

```bash
# List all configs        python train_configs.py --list
# Show one in detail      python train_configs.py --info nova
# Generate a runnable script:
python train_configs.py --generate coding > train_coding.py
python train_coding.py            # resumes/creates aira_model.pt
```

The direct way (equivalent, no wrapper):

```bash
python train.py --preset base  --data code_corpus.txt          # coding pass
python train.py --preset base                                 # daily pass
python train.py --preset nova                                 # flagship (BPE + RoPE)
```

Override anything from the CLI:

```bash
python train.py --preset nova --steps 60000 --vocab-size 8192 --tokenizer bpe --rope
python train.py --preset nova --fresh        # start over (ignore existing checkpoint)
```

---

## Auto-Resume: How the One Model Grows

Every run loads the existing `aira_model.pt` (exact filename is `DEFAULT_CHECKPOINT_NAME`).

- **Same architecture** (n_embd / n_head / n_layer / block_size **+ tokenizer kind + rope setting**
  all match) → training **resumes** from the saved step with the saved optimizer state.
- **Different architecture** (you switch presets) → the old checkpoint is kept as backup and a fresh
  model starts; your previous model file is preserved.
- Checkpoint contents: `{model, config, step, optimizer, tokenizer}`. In char mode the tokenizer
  blob is `chars`; in BPE mode it is the full merge table — model files carry their own vocabulary.

Example growth path (all into the same file):

```bash
python train.py --preset base    --data daily_conversations.txt   # 2.3M
python train.py --preset beast                                    # 10M (size switch: fresh start)
# Kaggle: keep re-attaching aira_model.pt and pressing run — steps add up
```

To force a clean restart inside the **same** architecture: `--fresh`.

---

## Hardware Guide

| Preset | Comfortable on |
|--------|----------------|
| `tiny`, `base` | Any laptop CPU |
| `beast` (10M) | Desktop CPU (a few minutes), any GPU instantly |
| `xl`, `xxl` | Free GPU strongly recommended |
| `nova` (91.3M) | Kaggle T4 (1× works, 2×T4 with DataParallel is faster) |
| `titan` (202M) | Kaggle P100/T4 — long sessions, mind the 12h cap |

Kaggle specifics (quota table, session-survival workflow, troubleshooting) live in
[docs/KAGGLE_TRAINING.md](KAGGLE_TRAINING.md). One-cell starters:

- `scripts/kaggle_one_cell.py` — paste-in single cell for a fresh notebook.
- `scripts/kaggle_dynamic_cell.py` — auto-detects 1×T4 vs 2×T4 / other GPUs and picks the preset,
  estimates wall time, and resumes from an attached previous output.

---

## Voice AI Training Notes

Voice AI (`voice_ai.py`, `voice_ai_girl.py`) loads the same `aira_model.pt` via
`ai_core.agent.load_model()` (returns `(model, codec)` — the codec handles both legacy char
checkpoints and BPE checkpoints transparently). There is no separate "voice model": run a
`daily`/`nova` pass and the improved brain is immediately in her voice modes.

- STT: Whisper (offline), TTS: pyttsx3; two-way interruption (`modules/interrupt.py`,
  `modules/tts_girl.py`) with an adaptive noise floor (`BargeInMonitor`) so random noise no longer
  falsely stops her.

---

## Training Tips

1. **Smoke-test first**: `python train.py --preset tiny --steps 20` proves the pipeline end-to-end.
2. **Watch the loss** — it should trend down; NaN means lower the learning rate or shrink the batch.
3. **BPE cost is one-time**: vocab 512 builds in ~11s on 391KB; 4096 takes a few minutes and is
   then stored in the checkpoint forever.
4. **Grow the corpus, then re-train**: Aira's ceiling at fixed params is set by data quality and
   diversity — target ≥100MB and keep adding passes (see KAGGLE_TRAINING.md, "Grow the corpus").
5. **Never delete `aira_model.pt` between Kaggle sessions** — re-attach last session's output as a
   dataset and the trainer continues from it.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Smaller preset, or rely on gradient accumulation (xxl/titan/nova already use it); on 2×T4 the batch is split across GPUs automatically |
| Resume says "architecture differs" | You switched presets/sizes — expected; the old file is kept and a new model starts |
| `PyAudio`/`sounddevice` install error | Install PortAudio system package, or run text-only mode |
| fp16 NaN on older GPUs | train.py auto-picks bfloat16 when supported; AMP + dropout already guard the rest |
| Slow char training on big corpora | Switch to `--tokenizer bpe` (nova does this by default) — ~6 chars per token means ~6× fewer positions |

---

## Next Steps

1. Smoke test: `python train.py --preset tiny --steps 20`
2. Real run: `python train.py --preset nova` (local GPU or Kaggle — use
   `scripts/kaggle_dynamic_cell.py`)
3. Talk to her: `python generate.py --prompt "Hello Aira"` or `python aira.py`

Full Kaggle walkthrough: [docs/KAGGLE_TRAINING.md](KAGGLE_TRAINING.md) ·
Voice guide: [docs/VOICE_AI_GUIDE.md](VOICE_AI_GUIDE.md) · Getting started:
[docs/QUICK_START.md](QUICK_START.md)

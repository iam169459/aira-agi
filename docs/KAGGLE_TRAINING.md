# Training Aira on Kaggle — the complete guide (free GPUs)

Scale the same from-scratch `EmotionAwareGPT` beyond `beast` (10M params)
using Kaggle's free GPU quota. No external model, no pretrained weights —
everything (architecture, tokenizer merges, emotion conditioning) is
initialized and trained from zero on your own corpus.

This guide covers everything current: the preset ladder (including
**nova**), the **byte-level BPE tokenizer**, **RoPE**, the one-model
growth loop, and the session-survival workflow that makes 12-hour
session limits irrelevant.

---

## 1. What Kaggle gives you for free

| Resource | Quota |
|---|---|
| **GPU T4 x2** (2 × 16 GB) | ~30 h/week — the default choice |
| **GPU P100** (1 × 16 GB) | same quota bucket |
| **TPU v3-8 / VM** | exist, but useless here — `train.py` is CUDA-only; pick T4 |
| Session length | 12 h per run (resumable — see §5) |
| Storage per notebook output | 20 GB (`/kaggle/working/`) |

Rules of thumb for the architecture math:
- T4 ×2 = 32 GB total VRAM. Mixed precision (built into `train.py`) lets
  `titan` (202M) train in ~7 GB.
- A 32B-parameter model would need ~500 GB — impossible on free tiers
  and pointless for a 0.4 MB corpus (see §7).

## 2. The preset ladder (all measured)

| Preset | Layers × dims (heads) | Context | Tokenizer | Params | Checkpoint | VRAM | Where it fits |
|---|---|---|---|---:|---:|---|---|
| `tiny`  | 3 × 96 (4)   | 64  | char | 0.36M | ~1.4 MB  | < 1 GB | CPU, seconds |
| `base`  | 5 × 192 (6)  | 128 | char | 2.3M  | ~8.7 MB  | ~1 GB | CPU, minutes |
| `beast` | 8 × 320 (8)  | 256 | char | 10M   | ~38 MB   | ~2 GB | CPU (slow) / any GPU |
| `xl`    | 10 × 512 (8) | 384 | char | 31.8M | ~121 MB  | ~3 GB | 1 T4, ~25 min/1k steps |
| `xxl`   | 12 × 768 (12)| 512 | char | 85.6M | ~326 MB  | ~5 GB | 1–2 T4, ~1.5 h/1k steps |
| `titan` | 16 × 1024 (16)| 512 | char | 202M | ~771 MB  | ~7 GB | 2 T4, ~3.5 h/1k steps |
| `nova`  | 12 × 768 (12)| 512 **BPE** | **BPE 4096 + RoPE** | 91.3M | ~350 MB | ~6 GB | 1–2 T4, ~1.2 h/1k steps |

- With 2 GPUs attached, `train.py` auto-wraps the model in
  `DataParallel` and splits the micro-batch across both T4s — no flags,
  no code changes.
- **`nova` is the modern flagship**: byte-level BPE (4096 vocab, learned
  from your corpus at train time — 3.63 chars/token measured vs 1.0
  char-level, so her 512-token context holds ~3.6× more text) plus RoPE
  positional encoding — with RoPE, generation is *not* capped at the
  training window (demonstrated: 350 tokens out of a 64-token window).

Recommendation: train **`xxl` or `nova`** for the current 0.4 MB corpus;
`nova` gives you the better architecture for the same param budget.

## 3. Choosing the switches

```bash
# legacy, classic presets (char tokenizer + learned positions):
python train.py --preset xxl

# add byte-level BPE (learned from OUR corpus, nothing downloaded):
python train.py --preset xxl --tokenizer bpe --vocab-size 4096

# add RoPE (Llama/Mistral-style rotary embedding, no context cap):
python train.py --preset xxl --tokenizer bpe --vocab-size 4096 --rope

# or just use the preset that has both by default:
python train.py --preset nova
```

Notes:
- BPE training happens first, in-process. For `--vocab-size 4096` on the
  full corpus expect a few minutes of pure-Python merge learning before
  step 0 — one-time cost, the merges travel inside the checkpoint.
- Old checkpoints keep working unchanged: default mode is char-level.
- Resume compatibility checks **architecture + tokenizer + positional
  scheme**. If `aira_model.pt` holds a char checkpoint,
  `--tokenizer bpe` raises a clear error telling you to use `--fresh`
  or `--out`. Deliberate, not a bug.

## 4. Click-by-click: from zero to a training session

1. Open https://www.kaggle.com → New Notebook.
2. **Settings → Accelerator → GPU T4 x2.**
3. **Settings → Internet → On** (only needed for the `git clone`).
4. Add the code cells below — or paste a ready-made cell:
   - **`scripts/kaggle_dynamic_cell.py`** — *recommended*: one cell that
     auto-detects 1x vs 2x T4 (GPU count + VRAM), picks the best preset
     automatically, prints a wall-clock estimate, and still honors the
     one-model resume loop.
   - [`scripts/kaggle_one_cell.py`](../scripts/kaggle_one_cell.py) —
     simpler single-cell option (fixed `nova` preset).
   - [`scripts/kaggle_train.ipynb`](../scripts/kaggle_train.ipynb) —
     multi-cell notebook.

**Cell 1 — sanity check:**
```python
import subprocess, sys, torch
n = torch.cuda.device_count()
print('GPUs:', n, [torch.cuda.get_device_name(i) for i in range(n)]
      or 'NONE — enable GPU in Settings!')
```

**Cell 2 — get the repo:**
```python
import subprocess, os
REPO = '/kaggle/working/aira-agi'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone',
                    'https://github.com/iam169459/aira-agi.git', REPO])
%cd {REPO}
```

**Cell 3 — train (one growing model):**
```python
PRESET = 'nova'          # or 'xxl', 'titan', ...
OUT = '/kaggle/working/aira_model.pt'
from train import train
train(PRESET, out_path=OUT)
```

**Cell 4 (when finished):**
```python
# Stop after ~8-10 GB / when you're ready; the checkpoint is at OUT.
print('Download from the Output tab:', OUT)
```

5. Confidence check before you leave the keyboard: the log should show
   `Device: cuda | GPUs: 2 (DataParallel) | AMP: torch.float16` and
   `Parameters: ...` matching §2. Loss prints at every eval interval,
   and each eval line is followed by a checkpoint save.

## 5. The session-survival workflow (how a 12 h cap never costs you work)

`train.py` was built for this exact situation:

1. **Periodic saves**: checkpoint (weights + optimizer + step counter)
   is written to `OUT` at every eval interval, not just at the end.
2. **Keep one model across sessions**: after a session ends, upload
   `aira_model.pt` as a **Kaggle Dataset** (Data → New Dataset → upload
   the .pt), attach it to the next notebook run, and copy it over the
   same path before training:

```python
OUT = '/kaggle/working/aira_model.pt'
import shutil, glob, os
prev = (glob.glob('/kaggle/input/*/aira_model.pt') or [None])[0] or \
       glob.glob('/kaggle/working/aira_model.pt')[0]
if prev and prev != OUT:
    shutil.copy(prev, OUT)
from train import train
train(PRESET, out_path=OUT)     # picks up weights+optimizer+step
```

3. Training resumes at the recorded global step and keeps growing the
   **same** model — run one session after another for as long as you
   like. This is the "one model" philosophy: you never end up with five
   half-trained checkpoints.
4. Watch for `Resuming ... at step N` in the log — if you see
   `Vocabulary`/tokenizer mismatch errors instead, you mixed char/BPE
   (or RoPE) checkpoints; use `--fresh` deliberately or a new `--out`.

## 6. Growing her — corpus first, then parameters

The wall-clock ladder for the current corpus:

```bash
python train.py --preset beast    # CPU-able sanity tier
python train.py --preset xl       # first Kaggle tier
python train.py --preset nova     # modern architecture tier
python train.py --preset xxl      # sweet spot for 0.4 MB corpus
python train.py --preset titan    # only after §6 corpus growth
```

To raise the corpus ceiling (currently ~0.4 MB → target ≥ 100 MB before
`titan` or `nova` long runs make sense):

1. Drop text files into `data/` (conversations, code, documentation —
   anything you own or have the rights to use).
2. Rebuild the combined corpus:

```python
from train import build_corpus
build_corpus()   # rewrites data/beast_corpus.txt from everything in data/
```

3. Resume the same preset — BPE merges and char vocab are re-learned
   against the new corpus on resume, and the model keeps its weights.

Because BPE packs 2.5–3.6× more text per token, `nova` also trains
~3× more *language* per optimizer step than the char presets —
the cheapest quality lever besides the corpus itself.

## 7. The honest limits (read before burning quota)

- **32B is not reachable and not meaningful here.** A 32B model needs
  ~500 GB VRAM (8×A100+) and a corpus of billions of tokens. Kaggle's
  free 2×T4 gives 32 GB and ~30 h/week. Aira's growth path is the
  measured one: corpus → `xxl`/`nova` → multi-session `titan`, then a
  1B target only when the corpus is ≥ 100 MB.
- **Overfitting is the real ceiling.** `data/` is ~0.4 MB of text;
  `titan` (202M) can memorize every byte of it. The honest signature:
  *train* loss → near zero while *val* loss stalls or rises. When that
  happens, more steps are waste — grow the corpus (§6) or stop.
- **Eval numbers, not vibes**: at every eval interval you get
  `{'train': ..., 'val': ...}`. Save checkpoints where val loss drops.

## 8. Troubleshooting

Early stopping (default `--patience 10`): training saves the checkpoint **only when the validation loss improves**; after 10 evals with no improvement the run ends early, restores the best weights, and `aira_model.pt` holds the best model — never an overfit one. `best_val` is stored inside the checkpoint so best-tracking continues across sessions.

| Symptom | Cause | Fix |
|---|---|---|
| `GPUs: 0` in Cell 1 | accelerator not enabled | Settings → GPU T4 x2, save |
| `CUDA out of memory` | preset too big for attached GPUs | use `--out` to branch, drop to `xxl`/`nova`, or free the other notebook |
| Resume raises model-size error | checkpoint from a different preset size | matching preset resumes; else `--fresh`/`--out` |
| Resume raises tokenizer error | char↔BPE checkpoint mismatch | same — deliberate switch, `--fresh` or `--out` |
| Resume raises `rope=` mismatch | mixed positional schemes | same pattern |
| Train loss 0.02, val loss flat/rising | overfitting (corpus < model) | automatic since PR #30: **early stopping** ends the run after 10 non-improving evals and keeps the BEST checkpoint (`--patience 0` disables; `--patience N` tunes). Real fix: §6 grow corpus |
| Session died mid-run | 12 h cap | §5: checkpoint is on disk at last eval; reattach + copy + rerun |
| Generation rambling/nonsense | model still young | more steps and/or corpus; try lower `--temperature` in `generate.py` |

---

*Kaggle is one option — the same commands run unchanged on any CUDA
machine (Colab free tier also works for `xl`/`xxl`/`nova`).*

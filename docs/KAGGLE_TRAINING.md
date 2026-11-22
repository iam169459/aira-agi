# Training Aira on Kaggle (free GPUs)

Scale the same from-scratch `EmotionAwareGPT` above `beast` (10M params)
using Kaggle's free GPU quota. No external model, no pretrained weights —
it is still 100% your own architecture, initialized and trained from zero.

## Preset ladder

| Preset | Layers × dims (heads) | Context | Params | Checkpoint (fp32) | VRAM while training |
|---|---|---|---|---|---|
| `tiny`  | 3 × 96 (4)   | 64  | 0.36M | ~1.4 MB  | < 1 GB (CPU ok) |
| `base`  | 5 × 192 (6)  | 128 | 2.3M  | ~8.7 MB  | ~1 GB (CPU ok)  |
| `beast` | 8 × 320 (8)  | 256 | 10M   | ~38 MB   | ~2 GB (CPU ok, slow) |
| `xl`    | 10 × 512 (8) | 384 | 31.8M | ~121 MB | ~3 GB (any GPU) |
| `xxl`   | 12 × 768 (12)| 512 | 85.6M | ~326 MB | ~5 GB (T4 fine) |
| `titan` | 16 × 1024 (16)| 512 | 202M  | ~771 MB | ~7 GB (T4 fine) |

All of `xl`/`xxl`/`titan` fit comfortably on **one** of Kaggle's free
T4s (16 GB), and `train.py` uses **both** T4s automatically via
`DataParallel` when two are attached (effective batch per optimizer
step is preserved — the micro-batch is split across GPUs). Rough
wall-clock per 1k optimizer steps on T4 x2: `xl` ~25 min, `xxl` ~1.5 h,
`titan` ~3.5 h.

## How to run on Kaggle

1. Zip this repo and upload it as a Kaggle Dataset, or let the notebook
   `git clone` it (cells in [`scripts/kaggle_train.ipynb`](../scripts/kaggle_train.ipynb)).
2. New Notebook → Settings → Accelerator → **GPU T4 x2** (or P100).
3. Internet → On (only needed for the clone).
4. Run the notebook, pick `PRESET = 'xxl'` (or `'titan'`), and let it run.
   Sessions cap at 12 h, but checkpoints save to `/kaggle/working/`
   continuously — re-run with `--steps` to continue on a fresh session.
5. Download `aira_model.pt` from the notebook's Output tab and drop it
   next to `aira.py` as `tiny_model.pt` (or pass `--out`).

Command line, on any CUDA machine (including Kaggle's terminal):

```bash
python train.py --preset xxl --out xxl_model.pt
python train.py --preset titan --out titan_model.pt
```

## What changed for GPU training

- `train()` now picks CUDA automatically (it previously only saw
  CPU/MPS — it would never have used a Kaggle GPU).
- Mixed precision on CUDA: bf16 on A100/T4-with-bf16 kernels,
  fp16 + `GradScaler` on T4. Cuts VRAM ~2× and speeds up T4 ~2-3×.
  The autocast lives inside the model's forward, because `DataParallel`
  worker threads do not inherit thread-local autocast state.
- Multi-GPU: with 2+ CUDA GPUs the model is wrapped in
  `DataParallel` and the micro-batch is scaled by device count, so
  checkpoints stay identical in format (`module.` prefixes are
  stripped before saving).
- Gradient accumulation (`grad_accum` per preset) so effective batch
  size scales without more VRAM — `titan` trains at an effective
  batch of 32 sequences × 512 context.

## The one honest warning

These presets are bigger than the corpus. `data/` holds ~0.4 MB of text;
`titan` (200M params) can memorize all of it. Evaluation loss will drop
to near zero on *train* while *val* loss stalls — that is overfitting,
not learning. Quality rules of thumb:

- `xl` / `xxl` are the sweet spot for the current dataset.
- Before training `titan`, grow `data/`: target ≥ 100 MB of text
  (conversations, code, anything you have rights to use). Drop files
  into `data/`, then rebuild:

```python
from train import build_corpus
build_corpus()  # rewrites data/beast_corpus.txt from everything in data/
```

"""Aira on Kaggle — everything in ONE code cell.

Paste this entire file as a single cell in a fresh Kaggle notebook
(or copy its contents). Before running:
    Settings -> Accelerator -> GPU T4 x2   (1x T4 also works)
    Settings -> Internet -> On (only needed for the git clone)

It:
  1. verifies GPUs (DataParallel turns on automatically with 2x T4),
  2. clones the repo - or syncs an existing clone to origin/main
     (fetch + hard reset; the repo dir holds no state, so stale old
     code from a crashed session can never run again),
  3. picks up a previous session's checkpoint (attached as a Kaggle
     Dataset) so your ONE model keeps growing across the 12h cap,
  4. prints a wall-clock estimate, then trains.

The config block below overrides anything: PRESET, STEPS, TOKENIZER
("char"/"bpe"), VOCAB_SIZE, ROPE. Leave the overrides at None to get the
preset's own choices (nova = BPE 4096 + RoPE).

Download aira_model.pt from the Output tab when done; re-running a later
session with the same settings resumes it (same architecture = continue;
different preset = fresh start, old file kept as backup).

Full guide: docs/KAGGLE_TRAINING.md · auto-sizing variant:
scripts/kaggle_dynamic_cell.py
"""
import glob
import os
import shutil
import subprocess
import sys

import torch

# ------------------------------------------------------------- config -----
REPO = "/kaggle/working/aira-agi"
OUT = "/kaggle/working/aira_model.pt"

PRESET = "nova"     # "nova" recommended | "titan" | "xxl" | "xl" | ...
STEPS = None        # None = preset default; set an int to limit this run
TOKENIZER = None    # None = preset choice | "bpe" | "char"
VOCAB_SIZE = None   # None = preset choice (nova: 4096)
ROPE = None         # None = preset choice | True | False

# Rough measured wall-clock on 1x T4, minutes per 1000 optimizer steps.
MINUTES_PER_1K = {"tiny": None, "base": None, "beast": 8, "xl": 25,
                  "xxl": 90, "nova": 72, "titan": 210}
HEAVY = {"xl", "xxl", "titan", "nova"}

# ------------------------------------------------------------- 1) GPUs ----
gpus = torch.cuda.device_count()
vram = [torch.cuda.get_device_properties(i).total_memory / 1e9
        for i in range(gpus)]
names = [torch.cuda.get_device_name(i) for i in range(gpus)]
print(f"GPUs: {gpus} "
      + (f"({', '.join(f'{a} {b:.1f}GB' for a, b in zip(names, vram))})"
         if gpus else
         "NONE — go to Settings → Accelerator → GPU T4 x2!"))
if gpus >= 2:
    print("DataParallel will split the batch across both GPUs "
          "(~1.8x faster).")

# ------------------------------------------------------------- 2) repo ----
# ALWAYS run the latest code: a fresh notebook gets a fresh clone, and a
# reused notebook is synced to origin/main (fetch + hard reset). A stale
# clone would silently rerun old, already-fixed code. The repo dir is
# disposable (all state lives in aira_model.pt), so a hard reset is safe.
HEAD = None
if os.path.exists(REPO):
    subprocess.run(["git", "-C", REPO, "fetch", "origin", "main"],
                   check=True)
    subprocess.run(["git", "-C", REPO, "reset", "--hard", "origin/main"],
                   check=True)
else:
    subprocess.run(["git", "clone",
                    "https://github.com/iam169459/aira-agi.git", REPO],
                   check=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
HEAD = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                      check=True, capture_output=True, text=True).stdout.strip()
print("Aira repo at commit:", HEAD)

# Drop modules imported from a PREVIOUS run in this same kernel
# (e.g. an earlier crashed cell): Python caches sys.modules, so without
# this purge 'from train import train' would return the OLD in-memory
# code even though the files on disk were just synced to latest.
import importlib
for _name in [m for m in sys.modules
              if getattr(sys.modules[m], "__file__", None)
              and str(sys.modules[m].__file__).startswith(REPO)]:
    del sys.modules[_name]
importlib.invalidate_caches()

from train import train, PRESETS   # noqa: E402

# --------------------------------------------------- 3) no-GPU guard ------
if gpus == 0 and PRESET in HEAVY:
    print(f"\nWARNING: preset '{PRESET}' on CPU would take days. "
          "Downgrading this run to 'tiny' so it still finishes — "
          "attach a GPU and keep PRESET='nova' for real training. "
          "(Set PRESET yourself to override.)")
    PRESET = "tiny"

# ------------------------------------------ 4) previous checkpoint --------
prev = glob.glob("/kaggle/input/*/aira_model.pt") or \
       ([OUT] if os.path.exists(OUT) else [])
if prev and prev[0] != OUT:
    shutil.copy(prev[0], OUT)
    print(f"Found previous checkpoint ({prev[0]}) — resuming your ONE "
          "model (architecture match) or starting fresh (backup kept).")
else:
    print("No previous checkpoint found — starting fresh.")

# ------------------------------------------------- 5) plan + estimate -----
preset_cfg = PRESETS.get(PRESET, {})
steps_for_estimate = STEPS or preset_cfg.get("max_steps", 0)
rate = MINUTES_PER_1K.get(PRESET)
if rate and steps_for_estimate:
    speedup = {1: 1.0, 2: 1.8}.get(min(gpus, 2), 1.0)
    hours = rate * (steps_for_estimate / 1000) / speedup / 60
    eta = f"≈ {hours:.1f} h"
else:
    eta = "minutes-class run"
plan = [f"preset={PRESET}", f"steps={steps_for_estimate}", eta]
if TOKENIZER or VOCAB_SIZE or ROPE is not None:
    tok = TOKENIZER or preset_cfg.get("tokenizer", "char")
    vocab = VOCAB_SIZE or preset_cfg.get("vocab_size")
    rope = preset_cfg.get("rope", False) if ROPE is None else ROPE
    plan.append(f"tokenizer={tok}")
    if vocab:
        plan.append(f"vocab={vocab}")
    plan.append(f"rope={'on' if rope else 'off'}")
print("Plan:", " | ".join(plan))

# ------------------------------------------------------------ 6) train ----
# Resumes automatically if the checkpoint's architecture matches; the
# overrides below (None values) defer to the preset's own choices.
kw = {k: v for k, v in dict(steps=STEPS, tokenizer=TOKENIZER,
                            vocab_size=VOCAB_SIZE, rope=ROPE).items()
      if v is not None}
train(PRESET, out_path=OUT, **kw)

# ------------------------------------------------------------- 7) done ----
print("\nFinished. Download from the Output tab:", OUT)
print("Next session: upload it as a Kaggle Dataset, attach it, run this "
      "cell again — she continues growing automatically.")

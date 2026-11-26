"""Aira on Kaggle — DYNAMIC one-cell trainer (adapts to 1x or 2x T4).

Paste this entire file as ONE cell in a fresh Kaggle notebook. It:
  1. reports every attached GPU + total VRAM,
  2. picks the best preset automatically (PRESET = "auto"), including
     DataParallel speedups when two T4s are attached (train.py does that
     for you — batch is split across GPUs automatically),
  3. prints a wall-clock estimate for your run,
  4. picks up a previous session's checkpoint (attached as a Dataset) so
     your ONE model keeps growing across the 12-hour session cap,
  5. trains.

Set PRESET manually ("titan"/"nova"/"xxl"/"xl"/"beast"/"base"/"tiny") to
skip auto-selection. Internet: On is needed only for the git clone.
Full guide: docs/KAGGLE_TRAINING.md
"""
import glob
import os
import shutil
import subprocess
import sys

import torch

REPO = "/kaggle/working/aira-agi"
PRESET = "auto"     # "auto" or a specific preset name
STEPS = None        # None = preset default; set an int to limit this run
OUT = "/kaggle/working/aira_model.pt"
FRESH = False       # True = ignore the attached checkpoint, start over
PATIENCE = None     # None = 10 evals | int | 0 disables early stopping

# Rough measured wall-clock on 1x T4, minutes per 1000 optimizer steps.
MINUTES_PER_1K = {"tiny": None, "base": None, "beast": 8, "xl": 25,
                  "xxl": 90, "nova": 72, "titan": 210}


def gpu_report():
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    vram = [torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(n)]
    print(f"GPUs: {n} "
          + (f"({', '.join(f'{a} {b:.1f}GB' for a, b in zip(names, vram))})"
             if n else "NONE — Settings → Accelerator → GPU T4 x2"))
    return n, sum(vram)


def pick_preset(gpus, total_vram_gb):
    """Choose the preset that fits this hardware best."""
    if gpus == 0:
        return ("tiny", "no GPU detected — tiny is the only CPU-sane tier; "
                        "re-run with a T4 attached for the real thing")
    if total_vram_gb < 5:
        return ("base", f"low VRAM ({total_vram_gb:.1f} GB total)")
    if total_vram_gb < 8:
        return ("xl", f"entry GPU ({total_vram_gb:.1f} GB total)")
    if gpus >= 2:
        return ("nova", f"{gpus} GPUs ({total_vram_gb:.1f} GB) + "
                        "DataParallel auto-on — nova is the sweet spot; "
                        "switch to PRESET='titan' once the corpus is big")
    return ("nova", "single 16 GB-class GPU — nova (BPE 4096 + RoPE) is "
                    "the recommended flagship")


def estimate(preset, gpus, steps):
    rate = MINUTES_PER_1K.get(preset)
    if rate is None or not steps:
        return "minutes-class run" if not rate else ""
    speedup = {1: 1.0, 2: 1.8}.get(min(gpus, 2), 1.0)   # DataParallel ~1.8x
    hours = rate * (steps / 1000) / speedup / 60
    return f"≈ {hours:.1f} h at this size on {gpus}GPU(s)"


# ------------------------------------------------------------------ setup --
# ALWAYS run the latest code: fresh notebook = fresh clone; reused
# notebook = fetch + hard reset to origin/main (the repo dir is
# disposable; your model state lives in aira_model.pt, not here).
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
print("Aira repo at commit:", subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], check=True,
    capture_output=True, text=True).stdout.strip())

# Drop modules imported from a PREVIOUS run in this same kernel
# (e.g. an earlier crashed cell): Python caches sys.modules, so without
# this purge both __import__("train") and 'from train import train'
# would return the OLD in-memory code even though the files on disk
# were just synced to latest.
import importlib
for _name in [m for m in sys.modules
              if getattr(sys.modules[m], "__file__", None)
              and str(sys.modules[m].__file__).startswith(REPO)]:
    del sys.modules[_name]
importlib.invalidate_caches()

gpus, vram_total = gpu_report()

if PRESET == "auto":
    PRESET, why = pick_preset(gpus, vram_total)
else:
    why = "manually selected"

prev = [] if FRESH else glob.glob("/kaggle/input/*/aira_model.pt")
if prev:
    shutil.copy(prev[0], OUT)
    print(f"Found previous checkpoint ({prev[0]}) — resuming your ONE model")
elif os.path.exists(OUT):
    print(f"Using existing {OUT} — resuming your ONE model")
else:
    print("No previous checkpoint found — starting fresh")

steps_for_estimate = (STEPS if STEPS else
                      __import__("train").PRESETS.get(PRESET, {}).get(
                          "max_steps", 0))
print(f"Plan: preset={PRESET} ({why}) | steps={steps_for_estimate} "
      f"| {estimate(PRESET, gpus, steps_for_estimate)}")

# ------------------------------------------------------------------ train --
# Early stopping keeps the BEST checkpoint: once val loss stops improving
# (memorization of a small corpus), the run ends and the best weights
# stay in aira_model.pt.
kw = {k: v for k, v in dict(steps=STEPS, patience=PATIENCE).items()
      if v is not None}
from train import train
train(PRESET, out_path=OUT, **kw)

print("\nFinished. Download from the Output tab:", OUT)
print("Next session: upload it as a Kaggle Dataset, attach it, run this "
      "cell again — she continues growing automatically.")

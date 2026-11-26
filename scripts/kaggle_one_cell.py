"""Aira on Kaggle — everything in ONE code cell.

Paste this entire file as a single cell in a fresh Kaggle notebook
(or copy its contents). Before running:
    Settings -> Accelerator -> GPU T4 x2
    Settings -> Internet -> On (only needed for the git clone)

It verifies GPUs, clones the repo, picks up a previous session's
checkpoint (attached as a Kaggle Dataset) so your ONE model keeps
growing, then trains. Download aira_model.pt from the Output tab when
done; re-running a later session with the same settings resumes it.

Full guide: docs/KAGGLE_TRAINING.md
"""
import subprocess, os, torch, glob, shutil

# 1) Sanity check: GPUs attached?
gpus = torch.cuda.device_count()
print("GPUs:", gpus, [torch.cuda.get_device_name(i) for i in range(gpus)]
      or "NONE — go to Settings → Accelerator → GPU T4 x2!")

# 2) Get the repo (Internet: On is needed just for this)
REPO = "/kaggle/working/aira-agi"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone",
                    "https://github.com/iam169459/aira-agi.git", REPO], check=True)
os.chdir(REPO)

# 3) Continue your ONE model across sessions (skips this on the first run)
OUT = "/kaggle/working/aira_model.pt"
prev = glob.glob("/kaggle/input/*/aira_model.pt") or \
       ([OUT] if os.path.exists(OUT) else [])
if prev and prev[0] != OUT:
    shutil.copy(prev[0], OUT)

# 4) Train — resumes automatically if the checkpoint already exists
#    PRESET: "nova" (recommended) | "xxl" | "titan" | "xl" | "beast" | ...
PRESET = "nova"
from train import train
train(PRESET, out_path=OUT)

# 5) Done — download aira_model.pt from the notebook's Output tab
print("\nFinished. Download from the Output tab:", OUT)
print("Next session: upload it as a Kaggle Dataset and run this cell again — "
      "she continues growing automatically.")

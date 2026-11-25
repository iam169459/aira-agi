# Aira AGI

**Aira is a personal AI assistant built from scratch — her own voice engine, her own language model, her own security toolkit.** No external AI APIs. Everything runs on your machine.

---

## What's inside

- **A brain that talks** — a custom emotion-aware transformer (`ai_core/`) with modern sampling controls, conversation memory that survives restarts, and always-on supervision so Aira never fully "sleeps".
- **A voice made from zero** — text-to-speech and speech-to-text built on pure numpy (`modules/natural_speech/`): grapheme-to-phoneme, glottal-pulse synthesis, prosody. She sounds human, and her voice changes with her mood.
- **A personality** — Aira answers in her own voice, picks up your mood, and tunes how creative or steady her answers are depending on it.
- **Hands for everyday tasks** — built-in tool skills (`ai_core/tools.py` + `ai_core/more_tools.py`): exact arithmetic, unit conversion, persistent notes, safe file reading, base64/JSON handling, hashing, secure password and UUID generation, dice/coin randomness, system status, and a full skills directory — just ask ("convert 5 km to miles", "generate a strong password 24", "what can you do").
- **A security toolkit** — defensive auditing plus authorized-target testing (`modules/cyber_tools/`), with a hacking knowledge base that teaches her what to look for and how to fix it.
- **Developer skills** — she can audit code and dependencies (`modules/dev_tools/`), and rewrite her own source through staged, audited, cancellable proposals (`ai_core/selfedit.py`).
- **Self-updating** — ask Aira to check for updates or update herself (`ai_core/updater.py`): she compares with `origin/main`, fast-forwards only, and refuses to touch uncommitted work. Fetched code loads on your next restart.

Every part has docs in [`docs/`](docs/), and every part has tests.

---

## Quick start

**Requirements:** Python 3.10+, PyTorch, NumPy. No internet needed to run.

```bash
pip install -r requirements.txt

python aira.py                     # interactive launcher
python voice_ai.py --mode voice    # talk to Aira out loud
python -m ai_core.daemon --with-agent   # keep her always on
```

Use her brain directly:

```python
from ai_core.agent import AiraAgent

aira = AiraAgent()
aira.think("hello aira!")
aira.think("convert 5 km to miles")   # grounded tool answer, no guessing
aira.think("check for updates")       # self-updater, fast-forward only
```

Extend her with a one-line plugin:

```python
from ai_core.tools import default_registry

@default_registry.tool(name="weather", patterns=[r"weather in (?P<arg>.+)"])
def weather(m):
    return f"{m.group('arg').title()}: sunny, obviously (offline joke)."
```

Train her (start small, go up). Every run grows the same single
checkpoint, `aira_model.pt` — later runs resume it automatically:

```bash
python train.py --preset tiny      # minutes
python train.py --preset base
python train.py --preset beast     # her full-size brain on CPU
python train.py --preset xxl       # ~86M params, needs a GPU
python train.py --preset titan     # ~200M params — see docs/KAGGLE_TRAINING.md for free Kaggle GPUs
```

---

## The one rule

Aira's security tools only touch targets **you own or have permission to test**. Loopback and private networks work out of the box; anything else needs your say-so (`--authorized` or a consent profile). She detects and repairs — she doesn't attack people. Details in [`docs/CYBER_TOOLS.md`](docs/CYBER_TOOLS.md).

---

## Testing

```bash
python -m pytest tests/ -q
```

## Updating

Aira updates herself from GitHub, safely:

```bash
# inside aira.py or the daemon, just say:
#   "check for updates"     -> read-only compare with origin/main
#   "update yourself"       -> fast-forward pull, refuses on dirty tree
# or from a shell, same code path:
python -m ai_core.updater
```

The full suite covers the voice engine, the transformer, the agent brain, the daemon, cybersecurity tooling, and developer skills.

---

## License & content protection

This project is **proprietary** — see [`LICENSE`](LICENSE). In particular:

- Training AI systems on this codebase, its docs, or its datasets is **explicitly prohibited**.
- Crawling for AI/ML use is opted out via [`robots.txt`](robots.txt) and [`.well-known/ai.txt`](.well-known/ai.txt).

Want the details on any subsystem? Everything is documented: [natural speech engine](docs/NATURAL_SPEECH.md) · [cybersecurity toolkit](docs/CYBER_TOOLS.md) · [developer skills](docs/DEV_SKILLS.md) · [AGI implementation](docs/AGI_IMPLEMENTATION.md).

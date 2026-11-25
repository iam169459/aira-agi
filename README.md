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
checkpoint, `aira_model.pt` — later runs resume it automatically, even
across crashed or time-capped sessions:

```bash
python train.py --preset tiny      # 0.36M params — minutes, CPU
python train.py --preset base      # 2.3M — CPU
python train.py --preset beast     # 10M — her full-size brain on CPU
python train.py --preset xl        # 31.8M — first GPU tier
python train.py --preset xxl       # 85.6M — Kaggle T4 sweet spot
python train.py --preset titan     # 202M — see docs/KAGGLE_TRAINING.md for free Kaggle GPUs

# Superpower switch: byte-level BPE tokenizer, learned from OUR OWN corpus
# (no pretrained tokenizer files). Packs ~2.5x more text into the same
# context window and can never fail on any input (emoji, CJK, anything).
python train.py --preset xxl --tokenizer bpe --vocab-size 4096
```

Drop more text into `data/`, rerun `python -c "import train;
train.build_corpus()"`, and keep training — the same model keeps
growing.

---

## The one rule

Aira's security tools only touch targets **you own or have permission to test**. Loopback and private networks work out of the box; anything else needs your say-so (`--authorized` or a consent profile). She detects and repairs — she doesn't attack people. Details in [`docs/CYBER_TOOLS.md`](docs/CYBER_TOOLS.md).

---

## Testing

The whole suite is green — **595 checks across 14 suites, 0 failures** —
covering the voice engine, the transformer, the tokenizer (char +
byte-level BPE), the agent brain, the tool skills, the daemon,
cybersecurity tooling, developer skills, and the two-way interruption
system:

| Suite | Checks | Suite | Checks |
|---|---:|---|---:|
| test_transformer | 18 | test_agi_power | 28 |
| test_aira_tools | 21 | test_agi_integration | 18 |
| test_more_tools | 31 | test_cyber_advanced | 38 |
| test_selfedit | 28 | test_cyber_professional | 50 |
| test_beast_mode | 92 | test_natural_speech | 42 |
| test_dev_skills | 55 | test_interrupt | 83 |
| test_cyber_tools | 48 | test_tokenizer | 43 |

```bash
python -m pytest tests/ -q        # if you have pytest
python -m tests.test_transformer  # or run any suite directly
python -m tests.test_more_tools
```

### How Aira compares

Full head-to-head with other AI models — sizes, architectures, what she
can and cannot do, and an honest verdict — lives in
[`comparison.md`](comparison.md). Short version: frontier models are
smarter by orders of magnitude (they are rented APIs trained on
trillions of tokens); Aira is the one that is fully yours — offline,
emotion-aware, tool-using, self-maintaining, and tested end to end.

## Updating

Aira updates herself from GitHub, safely:

```bash
# inside aira.py or the daemon, just say:
#   "check for updates"     -> read-only compare with origin/main
#   "update yourself"       -> fast-forward pull, refuses on dirty tree
# or from a shell, same code path:
python -m ai_core.updater
```

## Documentation

| Doc | Covers |
|---|---|
| [`QUICK_START.md`](docs/QUICK_START.md) | first-run walkthrough |
| [`KAGGLE_TRAINING.md`](docs/KAGGLE_TRAINING.md) | free-GPU training on the preset ladder |
| [`TRAINING_SUMMARY.md`](docs/TRAINING_SUMMARY.md) | training configs in depth |
| [`NATURAL_SPEECH.md`](docs/NATURAL_SPEECH.md) · [`VOICE_AI_GUIDE.md`](docs/VOICE_AI_GUIDE.md) | the voice engine |
| [`CYBER_TOOLS.md`](docs/CYBER_TOOLS.md) · [`EXAMPLE_PENTEST_REPORT.md`](docs/EXAMPLE_PENTEST_REPORT.md) | the security toolkit |
| [`DEV_SKILLS.md`](docs/DEV_SKILLS.md) | code audit, dependency hygiene, self-edit |
| [`AGI_IMPLEMENTATION.md`](docs/AGI_IMPLEMENTATION.md) | agent architecture |
| [`comparison.md`](comparison.md) | test results + how Aira compares to other AI models |
| [`GIRL_PERSONA_GUIDE.md`](docs/GIRL_PERSONA_GUIDE.md) · [`GIRL_VOICE_SUMMARY.md`](docs/GIRL_VOICE_SUMMARY.md) | persona & voice tuning |

---

## License & content protection

This project is **proprietary** — see [`LICENSE`](LICENSE). In particular:

- Training AI systems on this codebase, its docs, or its datasets is **explicitly prohibited**.
- Crawling for AI/ML use is opted out via [`robots.txt`](robots.txt) and [`.well-known/ai.txt`](.well-known/ai.txt).

Want the details on any subsystem? Everything is documented: [natural speech engine](docs/NATURAL_SPEECH.md) · [cybersecurity toolkit](docs/CYBER_TOOLS.md) · [developer skills](docs/DEV_SKILLS.md) · [AGI implementation](docs/AGI_IMPLEMENTATION.md).

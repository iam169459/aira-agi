# Quick Start Guide

## Train Aira AGI for Coding, Daily Usage, and Voice AI

---

## 🚀 Immediate Use (No Training Required)

### Start Voice AI
```bash
# Voice-to-voice conversation
python voice_ai.py --mode voice

# Text-to-voice (type, AI speaks)
python voice_ai.py --mode text

# Voice commands only
python voice_ai.py --mode command

# Interactive menu
```

---

## 🎯 Training for Specific Use Cases

Aira has ONE growing model — `aira_model.pt`. Every training command
resumes and grows the same file (auto-resume, periodic saves). Pick
whatever matches the use case:

### 1. Coding Assistant
```bash
python train.py --preset base --data code_corpus.txt
python generate.py --prompt "def aira_think():"
```

### 2. Daily Conversation AI
```bash
python train.py --preset base          # daily_conversations.txt
python generate.py --prompt "What should I cook?"
```

### 3. Voice AI Specialist (recommended)
```bash
python train.py --preset nova          # byte-level BPE 4096 + RoPE flagship
python voice_ai.py --mode voice        # picks up aira_model.pt automatically
```

### 4. Everything else — the measured ladder
```bash
python train.py --preset tiny   # 0.36M, seconds, CPU
python train.py --preset beast  # 10M,  CPU-able
python train.py --preset nova   # 91.3M — best quality, GPU tier
python train.py --preset titan  # 202M — Kaggle GPUs, see docs/KAGGLE_TRAINING.md

python train_configs.py --list   # convenience names + info
```

Full free-GPU playbook: `[KAGGLE_TRAINING.md](KAGGLE_TRAINING.md)`.

## 📚 Training Configurations

| Config | Use Case | Model Size | Time (CPU) | GPU |
|--------|----------|------------|------------|-----|
| `tiny` | Testing | Very Small | ~10 min | ❌ |
| `small` | Basic | Small | ~30 min | ⚠️ |
| `medium` | Balanced | Medium | ~1 hour | ✅ |
| `large` | Maximum | Large | ~2 hours | ✅ |
| `coding` | Code | Medium | ~1 hour | ✅ |
| `daily` | Conversation | Medium | ~45 min | ⚠️ |
| `voice` | Voice | Small | ~30 min | ⚠️ |

---

## 🎤 Voice AI Commands

### While in Voice Mode:
- **Speak naturally** after "Listening..." prompt
- **Say "goodbye" or "quit"** to end conversation
- **Press Ctrl+C** to stop at any time

### Supported Commands:
- "What time is it?"
- "What's the date?"
- "Tell me a joke"
- "What is [topic]?"
- "How do I [action]?"
- Any conversation topic

---

## 📁 New Files Created

```
tiny-from-scratch/
├── voice_ai.py                  # Voice-to-voice AI
├── train_configs.py             # Training configurations
├── modules/
│   └── aira_persona.py          # Girl persona module
├── examples/
│   └── demo_girl_persona.py     # Persona demo
└── data/
    ├── daily_conversations.txt      # Conversation dataset
    └── combined_training_data.txt   # Combined dataset
```

> **Guides:** see [VOICE_AI_GUIDE.md](VOICE_AI_GUIDE.md) and [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md)

---

## 🔧 Requirements

### Base Requirements (already in requirements.txt)
```bash
pip install -r requirements.txt
```

### Additional for Voice AI
```bash
pip install sounddevice scipy librosa soundfile
```

### Linux Audio Dependencies
```bash
sudo apt-get install portaudio19-dev python3-pyaudio espeak espeak-ng
```

### macOS Audio Dependencies
```bash
brew install portaudio espeak
```

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| PyAudio error | Install portaudio dependencies |
| No microphone | Check permissions and drivers |
| TTS not working | Install espeak (Linux) |
| CUDA out of memory | Use smaller model or batch size |
| Slow training | Use smaller config or GPU |

---

## 📖 Learn More

- **Complete Guide:** `VOICE_AI_GUIDE.md`
- **Training Details:** `TRAINING_SUMMARY.md`
- **AGI Implementation:** `AGI_IMPLEMENTATION.md`
- **Original Docs:** `README.md`

---

## 🎉 What You Can Do Now

### Without Training:
✅ Voice conversations
✅ Text-to-voice
✅ Voice commands
✅ Emotion detection
✅ AGI reasoning

### With Training:
✅ Coding assistant
✅ Daily conversation AI
✅ Voice specialist
✅ Custom models

---

**Start talking to your AI now!** 🚀

# Training Summary: Aira AGI for Coding, Daily Usage, and Voice AI

## What Was Created

This document summarizes all the new files and capabilities added to train Aira AGI for:
1. **Coding** - Code generation, understanding, and technical assistance
2. **Daily Usage** - General conversation, daily tasks, and personal assistance
3. **Voice-to-Voice AI** - Natural voice conversations with emotion detection

---

## New Files Created

### 1. Training Datasets

| File | Size | Purpose | Lines |
|------|------|---------|-------|
| `data/daily_conversations.txt` | 17KB | Daily conversation patterns | ~500 |
| `data/combined_training_data.txt` | 14KB | Combined code + conversation data | ~400 |

**Note:** The existing `data/code_corpus.txt` (383KB) is already available for coding training.

### 2. Voice AI System

| File | Purpose | Size |
|------|---------|------|
| `voice_ai.py` | Main voice-to-voice AI script | 26KB |
| `start_voice_ai.sh` | Interactive menu for voice AI | 2.4KB |

**Features:**
- Voice-to-voice conversation mode
- Text-to-voice mode
- Voice command mode
- Emotion detection from voice
- Emotion-aware responses
- Integration with AGI Core
- Young female voice preference for TTS

### 3. Training Configuration System

| File | Purpose | Size |
|------|---------|------|
| `train_configs.py` | Training configuration presets | 22KB |

**Includes 8 preset configurations:**
1. `tiny` - Minimal resources, quick testing
2. `small` - Basic capabilities, CPU-friendly
3. `medium` - Balanced performance
4. `large` - Maximum performance, GPU required
5. `coding` - Optimized for code generation
6. `daily` - Optimized for conversation
7. `voice` - Optimized for voice commands
8. `balanced` - General-purpose

### 4. Documentation

| File | Purpose | Size |
|------|---------|------|
| `VOICE_AI_GUIDE.md` | Complete guide for voice AI and training | 18KB |
| `TRAINING_SUMMARY.md` | This summary document | - |

---

## Quick Start Commands

### 1. Voice AI (No Training Required)

```bash
# Start voice conversation
python voice_ai.py --mode voice

# Start text conversation with voice output
python voice_ai.py --mode text

# Start voice command mode
python voice_ai.py --mode command

# Use interactive menu
chmod +x start_voice_ai.sh
./start_voice_ai.sh
```

### 2. Training for Coding

```bash
# Generate training script
python train_configs.py --generate coding > train_coding.py

# Run training (creates coding_model.pt)
python train_coding.py

# Use the trained model
python generate.py --checkpoint coding_model.pt --prompt "Write a Python function"
```

### 3. Training for Daily Conversation

```bash
# Generate training script
python train_configs.py --generate daily > train_daily.py

# Run training (creates daily_model.pt)
python train_daily.py

# Use the trained model
python generate.py --checkpoint daily_model.pt --prompt "What should I cook?"
```

### 4. Training for Voice AI

```bash
# Generate training script
python train_configs.py --generate voice > train_voice.py

# Run training (creates voice_model.pt)
python train_voice.py

# Use with voice AI
python voice_ai.py --mode voice
```

---

## Training Configuration Details

### Coding Configuration (`coding`)

**Purpose:** Optimized for code generation, understanding, and technical assistance

**Model Architecture:**
- Block size: 256 (for longer code snippets)
- Embedding dimension: 384
- Attention heads: 6
- Layers: 6
- Dropout: 0.1

**Training Parameters:**
- Batch size: 4
- Max steps: 2000
- Learning rate: 3e-4
- Eval every: 200
- Save every: 200

**Data Files:**
- Primary: `data/code_corpus.txt`
- Secondary: `data/combined_training_data.txt`

**Checkpoint:** `coding_model.pt`

**What it learns:**
- Code generation (Python, JavaScript, etc.)
- Code understanding and explanation
- Technical documentation
- Algorithm implementation
- Debugging assistance

### Daily Conversation Configuration (`daily`)

**Purpose:** Optimized for general conversation and daily assistance

**Model Architecture:**
- Block size: 128
- Embedding dimension: 256
- Attention heads: 4
- Layers: 4
- Dropout: 0.1

**Training Parameters:**
- Batch size: 8
- Max steps: 1500
- Learning rate: 3e-4
- Eval every: 150
- Save every: 150

**Data Files:**
- Primary: `data/daily_conversations.txt`
- Secondary: `data/combined_training_data.txt`

**Checkpoint:** `daily_model.pt`

**What it learns:**
- General conversation patterns
- Daily life questions and answers
- Personal assistance
- Everyday knowledge
- Natural language understanding

### Voice AI Configuration (`voice`)

**Purpose:** Optimized for voice interaction patterns

**Model Architecture:**
- Block size: 64 (voice commands are short)
- Embedding dimension: 128
- Attention heads: 4
- Layers: 3
- Dropout: 0.1

**Training Parameters:**
- Batch size: 16 (faster training)
- Max steps: 1000
- Learning rate: 5e-4
- Eval every: 100
- Save every: 100

**Data Files:**
- Primary: `data/combined_training_data.txt`

**Checkpoint:** `voice_model.pt`

**What it learns:**
- Voice command patterns
- Short, direct responses
- Conversational flow
- Quick reaction patterns

---

## Hardware Requirements

### For Voice AI (Inference Only)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4GB | 8GB+ |
| Storage | 500MB | 1GB+ |
| GPU | Not required | Optional (for faster STT) |
| OS | Linux/macOS/Windows | Linux/macOS/Windows |

### For Training

| Model Size | CPU | RAM | Storage | GPU | Time |
|------------|-----|-----|---------|-----|------|
| Tiny | 2 cores | 4GB | 1GB | No | ~10 min |
| Small | 4 cores | 8GB | 2GB | No | ~30 min |
| Medium | 4 cores | 8GB | 4GB | Recommended | ~1 hour |
| Large | 6+ cores | 16GB | 8GB | Required | ~2 hours |
| Coding | 4 cores | 8GB | 4GB | Recommended | ~1 hour |
| Daily | 4 cores | 8GB | 3GB | Optional | ~45 min |
| Voice | 4 cores | 8GB | 2GB | Optional | ~30 min |

---

## Voice AI Features

### 1. Speech-to-Text (STT)

- **Engine:** OpenAI Whisper
- **Models:** tiny, base, small, medium, large
- **Features:**
  - Real-time transcription
  - Multi-language support
  - Automatic punctuation
  - Speaker detection (with some models)

### 2. Text-to-Speech (TTS)

- **Engine:** pyttsx3
- **Features:**
  - Young female voice preference
  - Emotion-aware voice modulation
  - Adjustable rate and volume
  - Fallback to text output

### 3. Emotion Detection

- **Features:**
  - Audio-based emotion detection
  - MFCC feature extraction
  - Valence, Arousal, Dominance (VAD) model
  - Mood-adaptive responses

### 4. AGI Integration

- **Features:**
  - Full AGI Core integration
  - Reasoning engine
  - Memory system
  - Planning and decision making
  - World model
  - Meta-learning

---

## Usage Examples

### Example 1: Coding Assistant

```python
# After training with coding configuration

# Generate code
python generate.py --checkpoint coding_model.pt --prompt "Write a Python function to calculate factorial"

# With voice AI
python voice_ai.py --mode voice
# Say: "How do I write a for loop in Python?"
```

### Example 2: Daily Assistant

```python
# After training with daily configuration

# Generate conversation
python generate.py --checkpoint daily_model.pt --prompt "What should I cook for dinner?"

# With voice AI
python voice_ai.py --mode voice
# Say: "What's the weather like today?"
```

### Example 3: Voice Control

```python
# After training with voice configuration

# Voice commands
python voice_ai.py --mode command
# Say: "What time is it?"
# Say: "Tell me a joke"
# Say: "What's the date?"
```

---

## Training Tips

### 1. Start Small

```bash
# Test with tiny configuration first
python train_configs.py --generate tiny > train_tiny.py
python train_tiny.py
```

### 2. Monitor Progress

```bash
# Check training output for loss values
tail -f train_coding.py.log

# Look for:
# - Training loss should decrease
# - Validation loss should decrease or stabilize
# - No NaN or Inf values
```

### 3. Use Appropriate Hardware

```bash
# For CPU-only training, use smaller models
python train_configs.py --generate small > train_small.py
python train_small.py

# For GPU training, use larger models
python train_configs.py --generate medium > train_medium.py
python train_medium.py
```

### 4. Combine Datasets

```python
# Create a custom configuration with multiple datasets
from train_configs import TrainingConfig

config = TrainingConfig(
    name="Custom",
    data_files=["data/code_corpus.txt", "data/daily_conversations.txt", "data/combined_training_data.txt"],
    block_size=128,
    n_embd=256,
    n_head=4,
    n_layer=4,
    batch_size=8,
    max_steps=2000,
    checkpoint_file="custom_model.pt"
)
```

### 5. Save Checkpoints

```bash
# Training scripts automatically save checkpoints
# You can also manually save at any time
# Checkpoints are saved as: coding_model.pt, daily_model.pt, etc.
```

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| PyAudio installation error | Install portaudio dependencies |
| Whisper model download slow | Use `--stt-model tiny` for smaller model |
| No microphone detected | Check microphone permissions and drivers |
| TTS not working | Install espeak (Linux) or check system voices |
| CUDA out of memory | Reduce batch size or model size |
| Slow training on CPU | Use smaller model or reduce steps |

### Debug Commands

```bash
# Check available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test STT
python -c "from modules.stt import STTModule; stt = STTModule(model_size='tiny'); print(stt.list_available_models())"

# Test TTS
python -c "from modules.tts import TTSModule; tts = TTSModule(); tts.speak('Hello, world!')"

# Test AGI Core
python -c "from agi_core.integration import AGIAgent; agent = AGIAgent(); agent.initialize(); print(agent.converse('Hello'))"
```

---

## Best Practices

### For Training

1. **Start with small configurations** to test your setup
2. **Monitor loss values** to ensure training is progressing
3. **Use validation data** to check for overfitting
4. **Save checkpoints regularly** to avoid losing progress
5. **Experiment with different configurations** to find what works best

### For Voice AI

1. **Speak clearly** at a normal pace and volume
2. **Wait for prompts** before speaking
3. **Use short sentences** for better recognition
4. **Reduce background noise** for better accuracy
5. **Test your microphone** before starting

### For Deployment

1. **Choose appropriate model size** for your hardware
2. **Consider quantization** for faster inference
3. **Implement caching** for frequent responses
4. **Ensure privacy** when processing voice data
5. **Add authentication** for voice commands

---

## File Structure

```
tiny-from-scratch/
├── agi_core/                    # AGI Core modules
│   ├── integration/
│   │   └── agi_integration.py   # AGI integration
│   ├── reasoning/               # Reasoning engine
│   ├── memory/                  # Memory system
│   ├── planning/                # Planning system
│   ├── world_model/             # World model
│   ├── meta_learning/           # Meta-learning
│   └── nlp/                     # NLP modules
│
├── ai_core/                     # Multi-modal AI core
│   ├── main.py                  # Emotion-aware AI
│   └── model.py                 # Emotion-aware GPT model
│
├── modules/                     # Utility modules
│   ├── emotion.py               # Emotion detection
│   ├── stt.py                   # Speech-to-text
│   ├── tts.py                   # Text-to-speech
│   ├── screen_perception.py     # Screen perception
│   ├── desktop_automation.py    # Desktop automation
│   └── self_healing.py          # Self-healing
│
├── purple_ai/                   # Purple AI framework
│   └── ...
│
├── purple/                      # Extended Purple framework
│   └── ...
│
├── # NEW FILES #
│
├── voice_ai.py                  # Voice-to-voice AI script
├── start_voice_ai.sh            # Interactive menu
├── train_configs.py             # Training configurations
├── data/                        # Training datasets
├── VOICE_AI_GUIDE.md            # Complete guide
└── TRAINING_SUMMARY.md          # This summary
│
├── # EXISTING FILES #
│
├── train.py                     # Original training script
├── generate.py                  # Text generation script
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

---

## What You Can Do Now

### Without Training (Immediate Use)

✅ **Voice Conversation** - Talk to Aira with voice input and output
✅ **Text-to-Voice** - Type messages, get voice responses
✅ **Voice Commands** - Issue voice commands for quick actions
✅ **Emotion Detection** - Aira detects your mood from voice
✅ **AGI Integration** - Full AI reasoning and memory capabilities

### With Training (Enhanced Capabilities)

✅ **Coding Assistant** - Train for code generation and understanding
✅ **Daily Assistant** - Train for general conversation and tasks
✅ **Voice Specialist** - Train for voice interaction patterns
✅ **Custom Models** - Create your own specialized models

---

## Next Steps

1. **Try Voice AI immediately:**
   ```bash
   python voice_ai.py --mode voice
   ```

2. **Train for your use case:**
   ```bash
   # For coding
   python train_configs.py --generate coding > train_coding.py
   python train_coding.py
   
   # For daily conversation
   python train_configs.py --generate daily > train_daily.py
   python train_daily.py
   
   # For voice AI
   python train_configs.py --generate voice > train_voice.py
   python train_voice.py
   ```

3. **Explore AGI capabilities:**
   ```bash
   python -m agi_core.main --demo
   ```

4. **Read the full guide:**
   ```bash
   cat VOICE_AI_GUIDE.md
   ```

---

## Support and Resources

- **Documentation:** See `VOICE_AI_GUIDE.md` for detailed instructions
- **AGI Implementation:** See `AGI_IMPLEMENTATION.md` for technical details
- **Original README:** See `README.md` for project overview
- **Issues:** Open an issue on GitHub for problems or questions

---

## License

All code and datasets are licensed under the **MIT License**.

---

**Your AI is now ready for coding, daily usage, and voice-to-voice conversations!** 

Start exploring and training your personalized AI assistant today!

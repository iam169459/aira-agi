# Voice AI & Training Guide

> **Complete guide for training Aira AGI for coding, daily usage, and voice-to-voice AI**

---

## Table of Contents

- [Quick Start](#quick-start)
- [Voice AI Setup](#voice-ai-setup)
  - [Installation](#installation)
  - [Running Voice AI](#running-voice-ai)
  - [Voice Conversation Mode](#voice-conversation-mode)
  - [Text-to-Voice Mode](#text-to-voice-mode)
  - [Voice Command Mode](#voice-command-mode)
- [Training Your AI](#training-your-ai)
  - [Training Datasets](#training-datasets)
  - [Configuration Presets](#configuration-presets)
  - [Training for Coding](#training-for-coding)
  - [Training for Daily Usage](#training-for-daily-usage)
  - [Training for Voice AI](#training-for-voice-ai)
  - [Custom Training](#custom-training)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository (if not already done)
git clone https://github.com/iam169459/tiny-from-scratch.git
cd tiny-from-scratch

# Install Python dependencies
pip install -r requirements.txt

# Additional dependencies for voice
pip install sounddevice scipy librosa soundfile
```

### 2. Try Voice AI (No Training Required)

```bash
# Start voice conversation
python voice_ai.py --mode voice

# Or use the interactive menu
python aira.py
```

### 3. Train Your Model

```bash
# Train for coding
python train_configs.py --generate coding > train_coding.py
python train_coding.py

# Train for daily conversation
python train_configs.py --generate daily > train_daily.py
python train_daily.py

# Train for voice AI
python train_configs.py --generate voice > train_voice.py
python train_voice.py
```

---

## Voice AI Setup

### Installation

The Voice AI requires additional dependencies beyond the base project:

```bash
# Core dependencies (already in requirements.txt)
pip install torch openai-whisper pyttsx3 pyaudio

# Additional audio dependencies
pip install sounddevice scipy librosa soundfile

# For Linux (PyAudio dependencies)
sudo apt-get install portaudio19-dev python3-pyaudio

# For macOS
brew install portaudio
pip install pyaudio
```

### Running Voice AI

The `voice_ai.py` script provides three modes:

#### 1. Voice Conversation Mode (Full Voice-to-Voice)

```bash
python voice_ai.py --mode voice
```

**Features:**
- Speak naturally to Aira
- Aira responds with voice
- Emotion detection from your voice
- Emotion-aware responses
- AGI-powered intelligent conversation

**Usage:**
1. Run the command
2. When prompted "Listening...", speak your question or command
3. Aira will process your speech and respond with voice
4. Continue the conversation naturally
5. Say "goodbye", "bye", "quit", or "exit" to end

#### 2. Text-to-Voice Mode

```bash
python voice_ai.py --mode text
```

**Features:**
- Type your messages
- Aira responds with voice
- Good for testing voice output
- Useful in noisy environments

**Usage:**
1. Run the command
2. Type your message and press Enter
3. Aira will respond with voice
4. Type "quit", "exit", "goodbye", or "bye" to end

#### 3. Voice Command Mode

```bash
python voice_ai.py --mode command
```

**Features:**
- Single voice commands
- Quick execution
- Simple responses
- Good for voice control

**Usage:**
1. Run the command
2. Speak a command (e.g., "What time is it?")
3. Aira will execute and respond
4. Say "exit" or "quit" to end

### Voice AI Options

```bash
# Disable AGI Core (faster but less intelligent)
python voice_ai.py --mode voice --no-agi

# Disable STT (text input only)
python voice_ai.py --mode text --no-stt

# Disable TTS (text output only)
python voice_ai.py --mode voice --no-tts

# Disable emotion detection
python voice_ai.py --mode voice --no-emotion

# Use different Whisper model (tiny, base, small, medium, large)
python voice_ai.py --mode voice --stt-model small

# Adjust TTS settings
python voice_ai.py --mode voice --tts-rate 180 --tts-volume 0.8

# Use male voice (default is young female)
python voice_ai.py --mode voice --no-female
```

---

## Training Your AI

### Training Datasets

The project includes several training datasets:

1. **code_corpus.txt** (383KB)
   - Python, JavaScript, and other code examples
   - Technical documentation
   - Algorithm explanations
   - Best for: Coding assistant

2. **daily_conversations.txt** (17KB)
   - General conversation patterns
   - Daily life questions and answers
   - Personal assistance scenarios
   - Best for: Daily conversation AI

3. **combined_training_data.txt** (14KB)
   - Combination of code and conversation data
   - Voice interaction patterns
   - Multi-turn conversations
   - Best for: General-purpose AI

4. **data.txt** (Original)
   - Basic language model training data
   - Simple examples
   - Best for: Quick testing

### Configuration Presets

The `train_configs.py` module provides pre-configured training settings:

| Config | Use Case | Model Size | Training Steps | Best For |
|--------|----------|------------|----------------|----------|
| `tiny` | Testing | 64 dim, 2 heads, 2 layers | 500 | Quick testing, CPU-only |
| `small` | Basic | 128 dim, 4 heads, 3 layers | 1000 | Moderate resources |
| `medium` | Balanced | 384 dim, 6 heads, 6 layers | 3000 | Good performance |
| `large` | Maximum | 768 dim, 12 heads, 12 layers | 5000 | Best performance, GPU required |
| `coding` | Coding | 384 dim, 6 heads, 6 layers | 2000 | Code generation |
| `daily` | Conversation | 256 dim, 4 heads, 4 layers | 1500 | Daily conversation |
| `voice` | Voice AI | 128 dim, 4 heads, 3 layers | 1000 | Voice commands |
| `balanced` | General | 256 dim, 4 heads, 4 layers | 2000 | All-purpose |

### Training for Coding

To train a model optimized for coding assistance:

```bash
# Generate training script
python train_configs.py --generate coding > train_coding.py

# Run training
python train_coding.py

# The model will be saved as coding_model.pt
```

**What it learns:**
- Code generation (Python, JavaScript, etc.)
- Code understanding and explanation
- Technical documentation
- Algorithm implementation
- Debugging assistance

**Configuration details:**
- Block size: 256 (for longer code snippets)
- Embedding dimension: 384
- Attention heads: 6
- Layers: 6
- Training steps: 2000
- Batch size: 4

### Training for Daily Usage

To train a model optimized for daily conversation:

```bash
# Generate training script
python train_configs.py --generate daily > train_daily.py

# Run training
python train_daily.py

# The model will be saved as daily_model.pt
```

**What it learns:**
- General conversation patterns
- Daily life questions and answers
- Personal assistance
- Everyday knowledge
- Natural language understanding

**Configuration details:**
- Block size: 128
- Embedding dimension: 256
- Attention heads: 4
- Layers: 4
- Training steps: 1500
- Batch size: 8

### Training for Voice AI

To train a model optimized for voice interactions:

```bash
# Generate training script
python train_configs.py --generate voice > train_voice.py

# Run training
python train_voice.py

# The model will be saved as voice_model.pt
```

**What it learns:**
- Voice command patterns
- Short, direct responses
- Conversational flow
- Quick reaction patterns

**Configuration details:**
- Block size: 64 (voice commands are short)
- Embedding dimension: 128
- Attention heads: 4
- Layers: 3
- Training steps: 1000
- Batch size: 16 (faster training)

### Custom Training

To create a custom training configuration:

```python
from train_configs import TrainingConfig

# Create custom configuration
config = TrainingConfig(
    name="My Custom Model",
    description="Custom model for my specific needs",
    data_files=["code_corpus.txt", "daily_conversations.txt"],
    
    # Model architecture
    block_size=128,
    n_embd=256,
    n_head=4,
    n_layer=4,
    dropout=0.1,
    
    # Training parameters
    batch_size=8,
    max_steps=2000,
    learning_rate=3e-4,
    eval_every=200,
    
    checkpoint_file="my_model.pt"
)

# Save training script
from train_configs import save_training_script
save_training_script(config, "train_my_model.py")

# Run training
# python train_my_model.py
```

Or use the command line:

```bash
# Create a custom config file
cat > custom_config.py << 'EOF'
from train_configs import TrainingConfig

config = TrainingConfig(
    name="My Model",
    data_files=["code_corpus.txt", "daily_conversations.txt"],
    block_size=128,
    n_embd=256,
    n_head=4,
    n_layer=4,
    batch_size=8,
    max_steps=2000,
    checkpoint_file="my_model.pt"
)
EOF

# Generate and run
python -c "from custom_config import config; from train_configs import save_training_script; save_training_script(config)"
python train_my_model.py
```

---

## Hardware Requirements

### Minimum Requirements

| Task | CPU | RAM | Storage | GPU | Time |
|------|-----|-----|---------|-----|------|
| Voice AI (inference) | 2 cores | 4GB | 500MB | No | Real-time |
| Tiny model training | 2 cores | 4GB | 1GB | No | ~10 min |
| Small model training | 4 cores | 8GB | 2GB | No | ~30 min |
| Medium model training | 4 cores | 8GB | 4GB | Recommended | ~1 hour |
| Large model training | 6+ cores | 16GB | 8GB | Required | ~2 hours |

### Recommended Setup

**For Voice AI (inference only):**
- CPU: 2+ cores
- RAM: 4GB+
- Storage: 500MB+
- GPU: Not required
- OS: Linux, macOS, or Windows

**For Training (small models):**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 2GB+
- GPU: Optional (faster with GPU)
- OS: Linux, macOS, or Windows

**For Training (medium/large models):**
- CPU: 6+ cores
- RAM: 16GB+
- Storage: 8GB+
- GPU: NVIDIA CUDA GPU (recommended)
- OS: Linux (recommended), macOS, or Windows

### GPU Support

The training scripts automatically detect and use available GPUs:

- **CUDA**: NVIDIA GPUs on Linux/Windows
- **MPS**: Apple Silicon (M1/M2/M3) on macOS
- **CPU**: Fallback if no GPU available

To force a specific device:

```python
# In your training script, set:
device = "cuda"  # NVIDIA GPU
device = "mps"   # Apple Silicon
device = "cpu"   # CPU only
```

---

## Troubleshooting

### Common Issues

#### 1. PyAudio Installation Error

**Error:** `Could not build wheels for pyaudio`

**Solution:**
```bash
# On Linux
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# On macOS
brew install portaudio
pip install pyaudio

# On Windows
# Download pre-built wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio‑0.2.12‑cp310‑cp310‑win_amd64.whl
```

#### 2. Whisper Model Download

**Issue:** First run downloads Whisper model (can be large)

**Solution:**
- The base model is ~75MB
- Use `--stt-model tiny` for a smaller model (~39MB)
- Models are cached in `~/.cache/whisper`

#### 3. No Microphone Detected

**Error:** `No Default Input Device`

**Solution:**
```bash
# Check available devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Specify device manually
python voice_ai.py --mode voice
# Then modify the code to use the correct device ID
```

#### 4. TTS Not Working

**Issue:** No sound output

**Solution:**
```bash
# Check if pyttsx3 is working
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Hello'); engine.runAndWait()"

# If it fails, try:
# - On Linux: sudo apt-get install espeak espeak-ng
# - On macOS: brew install espeak
# - On Windows: Should work out of the box

# Fallback to text output
python voice_ai.py --mode voice --no-tts
```

#### 5. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
# In your training config, set batch_size=2 or batch_size=1

# Reduce model size
# Use a smaller configuration (tiny, small, or medium)

# Use gradient accumulation
# Modify training script to accumulate gradients over multiple batches
```

#### 6. Slow Training on CPU

**Issue:** Training is very slow on CPU

**Solution:**
- Use a smaller model configuration
- Reduce batch size
- Use fewer training steps
- Consider using a cloud GPU service

### Debug Mode

Enable verbose output for debugging:

```bash
# For voice AI
python voice_ai.py --mode voice 2>&1 | tee voice_debug.log

# For training
python train_coding.py 2>&1 | tee train_debug.log
```

---

## Advanced Usage

### Combining Multiple Models

You can train multiple models for different purposes and switch between them:

```python
# In generate.py, you can load different checkpoints
python generate.py --checkpoint coding_model.pt --prompt "Write a Python function to sort a list"
python generate.py --checkpoint daily_model.pt --prompt "What should I cook for dinner?"
```

### Transfer Learning

Fine-tune a pre-trained model on your specific data:

```python
# Start with a pre-trained model
import torch
from train import TinyGPT

# Load pre-trained model
checkpoint = torch.load('balanced_model.pt')
model = TinyGPT(len(checkpoint['chars']))
model.load_state_dict(checkpoint['model'])

# Continue training on new data
# ... (modify training loop to start from loaded model)
```

### Multi-GPU Training

For faster training with multiple GPUs:

```python
# In your training script
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
```

### Custom Voice Models

Train Whisper for your specific voice:

```bash
# Fine-tune Whisper on your voice samples
# This requires additional setup with Whisper fine-tuning
# See: https://github.com/openai/whisper#fine-tuning
```

### Emotion Detection Training

Improve emotion detection with your own voice samples:

```python
# Record voice samples with different emotions
# Label them with emotion (happy, sad, angry, neutral, etc.)
# Train a custom emotion classifier

# The emotion module uses MFCC features
# You can extend it with your own models
```

### Integration with AGI Core

The Voice AI integrates with the full AGI Core for advanced capabilities:

```python
from agi_core.integration import AGIAgent

# Create AGI agent
agent = AGIAgent(name="Aira", use_purple=True)
agent.initialize()

# Use in voice AI
response = agent.converse("What is artificial intelligence?")
print(response['response'])
```

### Custom Voice Commands

Add custom voice commands:

```python
# In voice_ai.py, extend the _process_command method

if 'play music' in command_lower:
    # Add your music playing logic
    return "Playing music..."

if 'turn on lights' in command_lower:
    # Add your smart home integration
    return "Turning on the lights..."
```

---

## Best Practices

### For Training

1. **Start small**: Begin with the `tiny` or `small` configuration to test your setup
2. **Monitor loss**: Check that the training loss is decreasing
3. **Use validation**: The validation loss should also decrease (or at least not increase)
4. **Save checkpoints**: Save models periodically in case of interruptions
5. **Experiment**: Try different configurations to find what works best for your use case

### For Voice AI

1. **Speak clearly**: Speak at a normal pace and volume
2. **Wait for prompts**: Wait for the "Listening..." prompt before speaking
3. **Short sentences**: Voice recognition works best with shorter sentences
4. **Quiet environment**: Reduce background noise for better recognition
5. **Test microphone**: Ensure your microphone is working properly

### For Deployment

1. **Model size**: Choose a model size that fits your hardware constraints
2. **Quantization**: Consider quantizing your model for faster inference
3. **Caching**: Cache frequent responses for faster performance
4. **Privacy**: Be aware of privacy implications when processing voice data
5. **Security**: Ensure voice commands are properly authenticated

---

## Example Workflows

### Workflow 1: Coding Assistant

```bash
# Train the coding model
python train_configs.py --generate coding > train_coding.py
python train_coding.py

# Use the coding model for generation
python generate.py --checkpoint coding_model.pt --prompt "Write a Python function to calculate Fibonacci numbers"

# Use with Voice AI
python voice_ai.py --mode voice
# Then ask: "How do I write a for loop in Python?"
```

### Workflow 2: Personal Assistant

```bash
# Train the daily conversation model
python train_configs.py --generate daily > train_daily.py
python train_daily.py

# Use with Voice AI
python voice_ai.py --mode voice
# Then ask: "What should I cook for dinner?"
```

### Workflow 3: Voice Control System

```bash
# Train the voice model
python train_configs.py --generate voice > train_voice.py
python train_voice.py

# Use voice command mode
python voice_ai.py --mode command
# Then say: "What time is it?"
```

### Workflow 4: Multi-Purpose AI

```bash
# Train the balanced model
python train_configs.py --generate balanced > train_balanced.py
python train_balanced.py

# Use with Voice AI
python voice_ai.py --mode voice
# Can handle coding, conversation, and voice commands
```

---

## Performance Tips

### Training Performance

- **Batch size**: Larger batch sizes use more memory but can be faster
- **Sequence length**: Longer sequences (block_size) capture more context but use more memory
- **Mixed precision**: Use `torch.autocast` for faster training on GPUs
- **Gradient checkpointing**: Reduce memory usage with gradient checkpointing

### Inference Performance

- **Model size**: Smaller models are faster for inference
- **Quantization**: Use 8-bit or 4-bit quantization for faster inference
- **Caching**: Cache token embeddings for repeated prompts
- **Batching**: Process multiple requests in parallel

### Voice Performance

- **Model size**: Use `tiny` or `base` Whisper model for faster transcription
- **Chunking**: Process audio in chunks for real-time transcription
- **Caching**: Cache TTS audio for frequent responses

---

## Contributing

If you improve the voice AI or training configurations, consider contributing back:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## License

All code is licensed under the MIT License. See the LICENSE file for details.

---

## Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Check the README.md for project information
- Review the AGI_IMPLEMENTATION.md for technical details

---

**Happy Training! Your AI is ready to learn and converse with you.** 

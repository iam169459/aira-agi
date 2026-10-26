# Aira AGI

> **A Complete Offline Artificial General Intelligence System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Offline](https://img.shields.io/badge/offline-100%25-brightgreen.svg)](https://shields.io/)
[![Privacy](https://img.shields.io/badge/privacy-100%25-important.svg)](https://shields.io/)
[![Status: Active](https://img.shields.io/badge/status-active-success.svg)](https://shields.io/)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Training](#training)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Components Deep Dive](#components-deep-dive)
- [Privacy & Security](#privacy--security)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Aira AGI** is a fully offline, private Artificial General Intelligence (AGI) system built from scratch. It integrates a decoder-only transformer architecture with advanced cognitive frameworks to deliver human-like reasoning, memory consolidation, goal-oriented planning, and adaptive learning—all executed entirely on local hardware without any internet connectivity or external API dependencies.

### What is AGI?

Artificial General Intelligence represents AI systems capable of general cognitive abilities comparable to human intelligence across diverse domains:

- **Reasoning & Problem Solving** - Logical deduction, causal reasoning, and multi-step problem decomposition
- **Learning & Adaptation** - Meta-learning capabilities that improve performance over time
- **Memory & Knowledge Retention** - Persistent, multi-tiered memory systems with consolidation
- **Planning & Goal Achievement** - Hierarchical planning with adaptive strategy selection
- **Natural Language Understanding** - Context-aware comprehension and generation
- **Perception & Action** - Multi-modal input processing and safe autonomous actions

Aira implements all of these capabilities in an integrated, lightweight, and fully local package designed for researchers, developers, and enthusiasts exploring AGI concepts.

---

## Key Features

### Core AGI Capabilities

| Capability | Description | Status |
|------------|-------------|--------|
| **Advanced Reasoning** | Chain-of-Thought (CoT), Tree-of-Thoughts (ToT), Reflection, Meta-Cognition, and Logical Deduction | ✅ |
| **Multi-Tier Memory** | Short-term (working), Long-term (semantic), Episodic memory with intelligent consolidation and retrieval | ✅ |
| **Hierarchical Planning** | Goal decomposition, action sequencing, constraint satisfaction, and adaptive replanning | ✅ |
| **World Model** | Dynamic knowledge graph with entities, relationships, spatial/temporal/causal reasoning | ✅ |
| **Meta-Learning** | Self-improvement engine, strategy optimization, architecture search, and curriculum learning | ✅ |
| **Natural Language Processing** | Intent classification, entity extraction, text processing, tokenization, and LLM integration | ✅ |
| **Emotion Awareness** | Valence-Arousal-Dominance (VAD) model for mood-adaptive, empathetic responses | ✅ |
| **Decision Engine** | Multi-criteria decision making with probabilistic reasoning and risk assessment | ✅ |
| **Auto Error Handling** | Self-healing system with intelligent error detection, diagnosis, and automatic recovery | ✅ |
| **Reflection Engine** | Self-evaluation, performance analysis, and continuous improvement | ✅ |

### Multi-Modal Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **Voice I/O** | OpenAI Whisper for speech-to-text (STT) and pyttsx3 for text-to-speech (TTS) | ✅ |
| **Screen Perception** | OpenCV-based screen capture and content analysis | ✅ |
| **Desktop Automation** | Mouse/keyboard control with PyAutoGUI, pynput, and comprehensive safety guards | ✅ |
| **Self-Healing Code** | Automatic error detection, analysis, and repair suggestions | ✅ |
| **Safe Mode Guardrails** | Configurable safety system blocking destructive or dangerous actions | ✅ |
| **Emotion Detection** | Audio-based emotion recognition and mood state tracking | ✅ |

### Transformer Model

| Feature | Description | Status |
|---------|-------------|--------|
| **From-Scratch GPT** | Custom decoder-only transformer implementation in PyTorch | ✅ |
| **Character-Level Tokenization** | Simple, efficient tokenization for training and generation | ✅ |
| **Multi-Head Attention** | Scaled dot-product attention with causal masking | ✅ |
| **Configurable Architecture** | Adjustable layers, heads, embedding dimension, and context size | ✅ |
| **Checkpoint Management** | Save and load model weights with full configuration | ✅ |

---

## Architecture

```
aira-agi/
├── README.md                    # Project documentation
├── AGI_IMPLEMENTATION.md        # Detailed technical implementation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── data.txt                     # Training corpus
├── train.py                     # Transformer training script
├── generate.py                  # Text generation script
├── test_agi_core.py             # Comprehensive test suite
├── __init__.py                  # Main entry point
├── agi_state.json               # Persistent AGI state (auto-generated)
│
├── agi_core/                    # AGI Core System (35+ modules)
│   ├── __init__.py              # Core exports (Aira, AGIAgent)
│   ├── main.py                  # AGI CLI interface
│   ├── integration/             # Unified integration layer
│   │   ├── __init__.py
│   │   └── agi_integration.py  # AGIAgent & AGIIntegration classes
│   ├── reasoning/               # Reasoning Engine
│   │   ├── __init__.py
│   │   ├── reasoning_engine.py # CoT, ToT, logical deduction
│   │   └── reflection_engine.py # Self-reflection & meta-cognition
│   ├── memory/                  # Memory System
│   │   ├── __init__.py
│   │   ├── memory_system.py    # Unified memory interface
│   │   ├── short_term.py       # Working memory
│   │   ├── long_term.py        # Semantic memory
│   │   └── episodic.py         # Episodic memory
│   ├── planning/                # Planning & Decision Making
│   │   ├── __init__.py
│   │   ├── planner.py          # Goal-based planning
│   │   └── decision_engine.py  # Multi-criteria decisions
│   ├── world_model/             # World Model
│   │   ├── __init__.py
│   │   └── knowledge_graph.py  # Knowledge representation
│   ├── meta_learning/           # Meta-Learning System
│   │   ├── __init__.py
│   │   ├── meta_learner.py     # Self-improvement
│   │   ├── self_improvement.py # Strategy optimization
│   │   ├── architecture_search.py
│   │   └── curriculum_learning.py
│   ├── nlp/                     # Natural Language Processing
│   │   ├── __init__.py
│   │   ├── text_processor.py   # Text processing
│   │   ├── tokenizer.py        # Tokenization
│   │   ├── intent_classifier.py # Intent detection
│   │   ├── entity_extractor.py # Named entity recognition
│   │   ├── llm_interface.py    # LLM interface
│   │   └── llm_interface_full.py # PyTorch LLM implementation
│   └── llm/                     # Transformer Integration
│       ├── __init__.py
│       └── transformer_integration.py
│
├── ai_core/                     # Multi-Modal AI Core
│   ├── __init__.py
│   ├── main.py                  # Emotion-aware AI assistant
│   └── model.py                 # EmotionAwareGPT model
│
├── modules/                     # Utility Modules
│   ├── __init__.py
│   ├── emotion.py               # Emotion feature extraction
│   ├── stt.py                   # Speech-to-text (Whisper)
│   ├── tts.py                   # Text-to-speech (pyttsx3)
│   ├── screen_perception.py     # Screen capture & analysis
│   ├── desktop_automation.py    # Desktop control with safety
│   └── self_healing.py          # Self-healing code system
│
├── purple_ai/                   # Purple AI Framework
│   ├── __init__.py
│   ├── main.py
│   ├── core/                    # Core brain & agent
│   ├── mood_engine/             # Mood & emotion engine
│   ├── memory/                  # Purple AI memory
│   ├── perception/              # Perception system
│   ├── reasoning/               # Reasoning modules
│   ├── safe_mode/               # Safety guardrails
│   └── tools/                   # Tool integration
│
└── purple/                      # Extended Purple Framework
    ├── __init__.py
    ├── brain.py
    ├── core/
    ├── modules/
    └── utils/
```

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **RAM**: Minimum 4GB, recommended 8GB+ (16GB+ for larger models)
- **Storage**: At least 2GB free space
- **GPU**: Optional but recommended for training (CUDA, MPS, or CPU supported)
- **OS**: Linux, macOS, or Windows

### Step 1: Clone Repository

```bash
git clone https://github.com/iam169459/aira-agi.git
cd aira-agi
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies Overview

Key packages from `requirements.txt`:
- `torch>=2.2` - Deep learning framework
- `torchvision` - Computer vision utilities
- `transformers` - Transformer utilities
- `numpy` - Numerical computing
- `openai-whisper` - Speech recognition
- `pyttsx3` - Text-to-speech
- `opencv-python` - Computer vision
- `pyautogui` - Desktop automation
- `pynput`, `keyboard`, `mouse` - Input control
- `fastapi`, `uvicorn` - API server (optional)
- `librosa` - Audio processing
- `psutil` - System monitoring

*Note: Some packages like `pyaudio` may require additional system dependencies. On Linux, install `portaudio19-dev`. On macOS, use Homebrew. On Windows, prebuilt wheels are usually available.*

---

## Quick Start

### Option 1: Run via Main Entry Point (Recommended)

```bash
python __init__.py
```

This will check dependencies, initialize all components, and launch the multi-modal AI assistant.

### Option 2: Run Aira AGI Core Directly

```bash
# Interactive demo showcasing all capabilities
python -m agi_core.main --demo

# Interactive conversation mode
python -m agi_core.main --conversation
```

### Option 3: Use as Python Library

```python
from agi_core.integration import AGIAgent

# Initialize Aira
agent = AGIAgent(name="Aira")
agent.initialize()

# Start conversation
response = agent.converse("Hello! What are your capabilities?")
print(response['response'])
```

---

## Usage Guide

### Aira AGI Agent API

The `AGIAgent` class is the primary interface to Aira's capabilities:

```python
from agi_core.integration import AGIAgent

agent = AGIAgent(name="Aira", use_purple=True)
agent.initialize()
```

#### Conversation
```python
result = agent.converse("Explain quantum entanglement")
print(result['response'])
print(f"Intent: {result['intent']['type']}")
print(f"Mood: {result['mood']}")
```

#### Reasoning
```python
from agi_core.reasoning import ThoughtChain

chain = agent.think("If all A are B and all B are C, are all A are C?", max_steps=8)
print(f"Conclusion: {chain.get_conclusion()}")
print(f"Confidence: {chain.confidence}")
for i, step in enumerate(chain.steps):
    print(f"Step {i+1}: {step.content}")
```

#### Memory Operations
```python
# Store information
mem_id = agent.remember("The capital of France is Paris", memory_type="fact")

# Recall information
results = agent.recall("France")
for item in results:
    print(item)
```

#### Planning
```python
plan_result = agent.plan("Learn Python programming in 30 days")
print(f"Plan ID: {plan_result['plan_id']}")
for i, action in enumerate(plan_result['actions']):
    print(f"{i+1}. {action['name']}: {action['description']}")
```

#### Decision Making
```python
decision = agent.decide(
    "Which programming language should I learn first?",
    ["Python", "JavaScript", "Rust", "Go"]
)
print(f"Recommended: {decision['chosen_option']}")
print(f"Reasoning: {decision['reasoning']}")
```

#### Learning
```python
agent.learn(
    concept="Recursion",
    definition="A function that calls itself to solve smaller instances of a problem",
    examples=["Factorial calculation", "Tree traversal"]
)
```

#### State Management
```python
# Get current state
state = agent.get_state()
print(f"Conversations: {state['conversation_count']}")
print(f"Memory stats: {state['memory_stats']}")

# Save state
agent.save()

# Shutdown
agent.shutdown()
```

### Emotion-Aware AI Core

```python
from ai_core.main import EmotionAwareAI

ai = EmotionAwareAI(model_name="base", use_gpu=False)
response = ai.converse("I'm feeling stressed today")
print(response)

# Voice mode
ai.start_voice_loop()
```

---

## API Reference

### AGIAgent Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__(name, use_purple)` | `name: str`, `use_purple: bool` | `AGIAgent` | Initialize Aira agent |
| `initialize()` | None | `None` | Initialize all components and load saved state |
| `converse(user_input)` | `user_input: str` | `Dict[str, Any]` | Process conversation with full AGI pipeline |
| `think(problem, max_steps)` | `problem: str`, `max_steps: int` | `ThoughtChain` | Use reasoning engine to solve problems |
| `remember(info, memory_type)` | `info: str`, `memory_type: str` | `str` | Store information in memory |
| `recall(query)` | `query: str` | `List[Any]` | Retrieve relevant memories |
| `plan(goal)` | `goal: str` | `Dict[str, Any]` | Create action plan for goal |
| `decide(description, options)` | `description: str`, `options: List[str]` | `Dict[str, Any]` | Make multi-criteria decision |
| `learn(concept, definition, examples)` | `concept: str`, `definition: str`, `examples: List[str]` | `str` | Learn new concept |
| `get_state()` | None | `Dict[str, Any]` | Get current AGI state |
| `save()` | None | `None` | Persist all state to disk |
| `shutdown()` | None | `None` | Save and shutdown gracefully |

### Response Format (converse)

```python
{
    'response': str,           # Generated response
    'intent': {
        'type': str,           # greeting/question/request/etc.
        'confidence': float
    },
    'mood': {
        'emotion': str,        # neutral/happy/sad/angry/excited/etc.
        'valence': float,
        'arousal': float,
        'dominance': float
    },
    'reasoning': dict,         # Reasoning chain details
    'memory_updates': list,    # Memory operation IDs
    'processing_time': float   # Processing time in seconds
}
```

---

## Training

### Train TinyGPT from Scratch

Aira includes a minimal character-level transformer for training on custom data.

#### Prepare Training Data

Edit `data.txt` with your training corpus (plain text):

```text
Your training data here...
Can be books, code, articles, or any text data.
```

#### Configure Training (`train.py`)

Default configuration (conservative for 16GB Mac):
```python
BLOCK_SIZE = 64      # Context window size
BATCH_SIZE = 8       # Batch size
MAX_STEPS = 500      # Training steps
N_EMBD = 96          # Embedding dimension
N_HEAD = 4           # Attention heads
N_LAYER = 3          # Transformer layers
LEARNING_RATE = 3e-4
```

For better results, increase these based on available hardware.

#### Run Training

```bash
python train.py
```

Training will output:
- Device used (cuda/mps/cpu)
- Vocabulary size
- Model parameters count
- Loss every `EVAL_EVERY` steps
- Final checkpoint saved as `tiny_model.pt`

#### Generate Text

```bash
python generate.py
```

Or use the model programmatically:
```python
import torch
from train import TinyGPT

checkpoint = torch.load('tiny_model.pt')
model = TinyGPT(len(checkpoint['chars']))
model.load_state_dict(checkpoint['model'])
# ... generate
```

### Train for Coding on Kaggle

For coding-specific models, see the comprehensive guide in the repository or follow these steps:

1. **Prepare Dataset**: Collect code files (.py, .js, .ts, .cpp, etc.) into a corpus
2. **Upload to Kaggle**: Create dataset with your code corpus
3. **Use GPU**: Enable T4/P100 GPU in Kaggle notebook
4. **Update Config**: Increase `BLOCK_SIZE` to 256, `BATCH_SIZE` to 16, `N_LAYER` to 6+
5. **Train**: Run modified `train.py` for 5000+ steps
6. **Download Checkpoint**: Save `aira_coder.pt` from `/kaggle/working/`

See `AGI_IMPLEMENTATION.md` for detailed Kaggle training instructions.

---

## Configuration

### Environment Variables

Aira works offline by default. No API keys required. Optional configurations can be set in code.

### Model Configuration

Modify in `train.py`:
```python
BLOCK_SIZE = 256     # Context length
N_EMBD = 384         # Model dimension
N_HEAD = 6           # Number of heads
N_LAYER = 6          # Number of layers
DROPOUT = 0.2        # Dropout rate
```

Larger configurations require more RAM/VRAM:
- Small (96/4/3): ~1M params, runs on CPU
- Medium (256/8/6): ~10M params, needs GPU recommended
- Large (512/8/12): ~50M+ params, needs strong GPU

### AGI Agent Configuration

```python
agent = AGIAgent(
    name="Aira",      # Agent name
    use_purple=True   # Enable Purple AI integration
)
```

---

## Testing

### Run All Tests

```bash
python test_agi_core.py
```

### Test Output

The test suite covers:
- AGI Agent initialization
- Conversation system
- Reasoning engine
- Memory system (STM/LTM/Episodic)
- Learning system
- Planning system
- Decision making
- State management

Expected output:
```
======================================================================
  AGI Core Comprehensive Test Suite
======================================================================
...
   ✅ PASS: AGI Agent initialization
   ✅ PASS: Conversation
   ✅ PASS: Reasoning engine
   ...
   Total: 1/1 test suites passed
   Time: X.XX seconds
```

### Manual Testing

```python
# Test specific components
from agi_core.reasoning import ReasoningEngine
from agi_core.memory import MemorySystem
from agi_core.planning import Planner

# Test reasoning
re = ReasoningEngine()
chain = re.chain_of_thought("Solve this problem...")
print(chain.get_conclusion())
```

---

## Project Structure

### Core Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `agi_core/integration` | Unified AGI interface | `AGIAgent`, `AGIIntegration` |
| `agi_core/reasoning` | Logical reasoning | `ReasoningEngine`, `ThoughtChain`, `ReflectionEngine` |
| `agi_core/memory` | Memory management | `MemorySystem`, `ShortTermMemory`, `LongTermMemory`, `EpisodicMemory` |
| `agi_core/planning` | Goal planning & decisions | `Planner`, `Goal`, `Action`, `DecisionEngine` |
| `agi_core/world_model` | Knowledge representation | `WorldModel`, `KnowledgeGraph`, `Entity`, `Relationship` |
| `agi_core/meta_learning` | Self-improvement | `MetaLearner`, `SelfImprovementEngine`, `ArchitectureSearch` |
| `agi_core/nlp` | Language processing | `TextProcessor`, `IntentClassifier`, `EntityExtractor`, `LLMInterface` |

### Multi-Modal Modules

| Module | Purpose |
|--------|---------|
| `ai_core/main.py` | Emotion-aware AI with voice support |
| `modules/emotion.py` | Emotion feature extraction (MFCC-based) |
| `modules/stt.py` | Whisper speech recognition |
| `modules/tts.py` | pyttsx3 text-to-speech |
| `modules/screen_perception.py` | Screen capture and OCR |
| `modules/desktop_automation.py` | Safe desktop control |
| `modules/self_healing.py` | Automatic error repair |

---

## Components Deep Dive

### Reasoning Engine
Implements multiple reasoning strategies:
- **Chain-of-Thought**: Step-by-step logical reasoning
- **Tree-of-Thoughts**: Branching exploration of solution paths
- **Reflection**: Self-evaluation and error correction
- **Meta-Cognition**: Thinking about thinking

### Memory System
Multi-tiered architecture:
- **Short-Term Memory (STM)**: Limited capacity working memory (7±2 items)
- **Long-Term Memory (LTM)**: Persistent semantic knowledge with categorization
- **Episodic Memory**: Time-stamped experiences with emotional context
- **Consolidation**: Automatic transfer from STM to LTM during idle periods

### Planning System
Hierarchical goal decomposition:
- **Goal Creation**: Define objectives with priorities and constraints
- **Action Planning**: Generate step-by-step action sequences
- **Plan Execution**: Track progress and adapt to changes
- **Decision Making**: Evaluate options using multi-criteria analysis

### World Model
Knowledge representation:
- **Entities**: Objects, concepts, people, places
- **Relationships**: Connections between entities with types and weights
- **Properties**: Attributes and characteristics
- **Queries**: Graph traversal and inference

---

## Privacy & Security

### Privacy Features
- **100% Offline Operation**: No network calls, no cloud dependencies
- **Local Data Storage**: All memories, states, and models stored locally
- **No Telemetry**: Zero data collection or analytics
- **No External APIs**: Complete self-contained system
- **User Control**: Full control over all data

### Security Features
- **Safe Mode Guardrails**: Prevents dangerous desktop automation actions
- **Self-Healing System**: Detects and mitigates errors safely
- **Input Validation**: Sanitizes all user inputs
- **Local Execution Only**: Sandboxed to local environment
- **No Privilege Escalation**: Runs with user permissions

### Data Storage
- `agi_state.json` - Conversation history and AGI state
- `tiny_model.pt` - Trained transformer checkpoint
- Memory systems persist automatically via `agent.save()`

---

## Performance

### Resource Requirements

| Configuration | RAM | CPU | GPU | Use Case |
|---------------|-----|-----|-----|----------|
| Minimal (Default) | 4GB | 2 cores | None | Basic conversation, reasoning |
| Recommended | 8GB | 4 cores | Optional | Full features, smooth operation |
| Training | 16GB+ | 6+ cores | CUDA/MPS | Model training from scratch |

### Performance Characteristics
- **Response Time**: 100-500ms for typical conversations (CPU)
- **Reasoning**: 1-3 seconds for complex multi-step reasoning
- **Memory Queries**: Near-instantaneous
- **Training**: ~1-2 steps/second on CPU, 10-50 steps/second on GPU
- **Model Size**: Default transformer ~2-3M parameters (~10-15MB)

### Optimization Tips
- Use GPU acceleration for training and generation
- Reduce `BLOCK_SIZE` if memory constrained
- Disable unused components (e.g., `use_purple=False`)
- Close other applications during training

---

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Missing dependencies
pip install -r requirements.txt

# Check specific module
python -c "from agi_core.integration import AGIAgent; print('OK')"
```

#### PyAudio Installation Issues (Linux)
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

#### PyAudio Issues (macOS)
```bash
brew install portaudio
pip install pyaudio
```

#### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `train.py`
- Reduce `BLOCK_SIZE`
- Use smaller model configuration (`N_EMBD`, `N_LAYER`)
- Use gradient accumulation

#### Whisper Model Download
Whisper models download automatically on first use. Ensure sufficient disk space (~75MB-3GB depending on model size).

#### TTS Not Working
On Linux, install espeak:
```bash
sudo apt-get install espeak espeak-ng
```
The system falls back to text output if TTS unavailable.

#### EOF Errors in Non-Interactive Mode
The system handles EOF gracefully with auto-error recovery.

### Debug Mode

Enable verbose logging by modifying source files or using try/except blocks to see detailed errors.

---

## Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute
- **Bug Reports**: Open an issue with detailed reproduction steps
- **Feature Requests**: Suggest new capabilities or improvements
- **Code Contributions**: Submit pull requests with enhancements
- **Documentation**: Improve README, docs, or code comments
- **Testing**: Add test cases or improve coverage
- **Training Data**: Share curated datasets (respecting licenses)

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following existing code style
4. Test thoroughly (`python test_agi_core.py`)
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to classes and functions
- Keep functions focused and modular
- Use type hints where appropriate
- Maintain backward compatibility

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024-2026 Aira AGI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See [LICENSE](LICENSE) file for full details.

---

## Acknowledgments

- **Transformer Architecture**: Inspired by Vaswani et al.'s "Attention Is All You Need"
- **GPT Models**: Built on concepts from OpenAI's GPT architecture
- **Cognitive Science**: Informed by research in memory, reasoning, and planning
- **Open Source Community**: Thanks to all contributors and the amazing open-source ecosystem
- **PyTorch Team**: For the excellent deep learning framework
- **OpenAI Whisper**: For robust speech recognition

---

## Links

- **Repository**: [https://github.com/iam169459/aira-agi](https://github.com/iam169459/aira-agi)
- **Issues**: [https://github.com/iam169459/aira-agi/issues](https://github.com/iam169459/aira-agi)
- **Discussions**: [https://github.com/iam169459/aira-agi/discussions](https://github.com/iam169459/aira-agi)

---

**Built with ❤️ for the future of accessible, private AGI**

*Aira - Your fully offline Artificial General Intelligence companion*

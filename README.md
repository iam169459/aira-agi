# Aira AGI

> **A Complete Offline Artificial General Intelligence System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary--no--AI--training-red.svg)](LICENSE)
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
- [Content & AI Protection](#content--ai-protection)
- [License](#license)
- [Components Deep Dive](#components-deep-dive)
- [Cybersecurity Tools](#cybersecurity-tools-defensive--authorized-target)
- [Privacy & Security](#privacy--security)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
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
| **Voice I/O** | Whisper/pyttsx3 with an automatic from-scratch fallback engine: formant-synthesis TTS + MFCC/DTW speech recognition (no external APIs or models) | ✅ |
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
| **Multi-Head Attention** | Scaled dot-product attention with causal masking, fused into a single QKV projection | ✅ |
| **KV-Cache Generation** | Incremental decoding with a per-layer key/value cache (~4.4× faster than full recomputation) | ✅ |
| **Emotion Conditioning** | `EmotionAwareGPT` shifts generation by an 8-class emotion embedding | ✅ |
| **Configurable Architecture** | Adjustable layers, heads, embedding dimension, and context size | ✅ |
| **Checkpoint Compatibility** | Legacy per-head checkpoints remap to the fused layout on load | ✅ |

---

## Architecture

```
aira-agi/
├── README.md                    # Project documentation
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
│   ├── main.py                  # Emotion-aware AI assistant (lazy model init)
│   ├── model.py                 # EmotionAwareGPT model (shared by train/generate)
│   ├── agent.py                 # AiraAgent: model-backed brain + conversation memory
│   └── daemon.py                # Always-on supervisor: duties, heartbeat, self-restart
│
├── modules/                     # Utility Modules
│   ├── __init__.py
│   ├── emotion.py               # Emotion feature extraction
│   ├── stt.py                   # Speech-to-text (Whisper, w/ natural fallback)
│   ├── tts.py                   # Text-to-speech (pyttsx3, w/ natural fallback)
│   ├── natural_speech/          # From-scratch voice engine — see docs/NATURAL_SPEECH.md
│   │   ├── normalize.py         #   text normalization
│   │   ├── g2p.py               #   grapheme-to-phoneme
│   │   ├── phonemes.py          #   phoneme tables
│   │   ├── tts.py + synth.py    #   formant-synthesis TTS
│   │   ├── features.py + dtw.py #   MFCC + vectorized DTW matching
│   │   └── stt.py               #   vocabulary STT with self-enrollment
│   ├── screen_perception.py     # Screen capture & analysis
│   ├── desktop_automation.py    # Desktop control with safety
│   ├── self_healing.py          # Self-healing code system
│   └── cyber_tools/             # Cybersecurity toolkit — see docs/CYBER_TOOLS.md
│       ├── guard.py             #   authorization gate for active scans
│       ├── consent.py           #   consent profiles: pre-approved own targets
│       ├── hashing.py           #   digests, hash ID, password strength, baselines
│       ├── crypto.py            #   SHA-256-CTR + HMAC encryption, PBKDF2
│       ├── portscan.py          #   concurrent TCP scan with banners
│       ├── webaudit.py          #   HTTP security-header audit, TLS expiry
│       ├── logscan.py           #   IOC extraction, brute-force detection
│       ├── recon.py             #   DNS lookup, reverse DNS, subdomain sweep
│       ├── crack.py             #   authorized hash/credential auditing
│       ├── wall.py              #   cyber-wall crack detection + TLS versions
│       ├── webpwn.py            #   non-destructive injection probes
│       ├── repair.py            #   finding-to-fix remediation engine
│       ├── sentinel.py          #   live tamper/rogue-port/brute-force watch
│       ├── engagement.py        #   attack + repair pipeline in one pass
│       ├── fingerprint.py       #   web-tech fingerprint, dir probe, banner age, cipher audit
│       ├── reporting.py         #   severity-ranked Markdown/HTML pentest reports
│       └── clitool.py           #   python -m modules.cyber_tools.clitool
│
├── tests/                       # Test suites (pytest-compatible)
│   ├── test_natural_speech.py   #   TTS/STT engine roundtrip & DTW invariants
│   ├── test_transformer.py      #   fused attention, KV cache, legacy remap
│   ├── test_cyber_tools.py      #   hashes, crypto, scanners, guard, log parsing
│   ├── test_cyber_advanced.py   #   crack, wall, probe, repair, sentinel, engage
│   ├── test_cyber_professional.py  # fingerprint, dir probe, banners, ciphers, reports
│   ├── test_agi_core.py         #   AGI core components
│   ├── test_agi_power.py        #   sampling, agent brain, daemon
│   ├── test_agi_integration.py  #   unified brain wiring, memory persistence
│   ├── test_beast_mode.py       #   consent profiles, persona, emotion prosody
│   └── ...
│
├── docs/                        # Guides & examples
│   ├── NATURAL_SPEECH.md        #   from-scratch TTS/STT engine guide
│   ├── CYBER_TOOLS.md           #   cybersecurity toolkit guide
│   ├── EXAMPLE_PENTEST_REPORT.md #  sample severity-ranked pentest report
│   └── AGI_IMPLEMENTATION.md    #   AGI architecture & training notes
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

### Option 1: Run via Unified Launcher (Recommended)

```bash
python aira.py
```

An interactive menu for every mode: girl-persona voice chat, classic assistant, persona demo, training, generation, and tests. Shortcuts also work directly:

```bash
python aira.py girl     # girl voice conversation
python aira.py text     # girl text conversation
python aira.py demo     # persona demo
python aira.py test     # AGI core test suite
```

### Option 2: Run via Main Entry Point

```bash
python scripts/run_assistant.py
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

Edit `data/data.txt` with your training corpus (plain text):

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

See `docs/AGI_IMPLEMENTATION.md` for detailed Kaggle training instructions.

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
python -m pytest tests/ -q
```

Or run each suite standalone (also works under pytest):

```bash
python tests/test_natural_speech.py   # TTS/STT engine: roundtrips, DTW, WAV I/O
python tests/test_transformer.py      # fused attention, KV cache, legacy remap
python tests/test_agi_power.py        # sampling controls, agent brain, always-on daemon
python tests/test_agi_integration.py  # unified brain wiring, LLM bridge, memory persistence
python tests/test_agi_core.py         # AGI core components
python tests/test_cyber_tools.py      # hashes, crypto, scanners, guard, logs
python tests/test_cyber_advanced.py   # crack/wall/probe/repair/sentinel/engage
python tests/test_cyber_professional.py  # fingerprint, dirs, banners, ciphers, reports
```

### Test Output

The test suites cover:
- **Natural speech engine** — text normalization, G2P, TTS→STT word roundtrips,
  multi-word decoding, OOV rejection, enrollment, WAV I/O, phrase cache, DTW invariants
- **Cybersecurity toolkit** — digest known-answer vectors, hash ID, password
  strength, integrity baselines, crypto roundtrip + tamper rejection, loopback
  port scan, brute-force/IOC log parsing, live local HTTP header audit
- **Transformer** — fused QKV vs per-head equivalence, KV-cached vs recomputed
  generation, fallback past `block_size`, legacy checkpoint remap, emotion conditioning,
  training loss; generation benchmark (informational)
- **AGI power + integration** — sampling controls, agent brain, always-on daemon,
  unified brain wiring, LLM bridge, memory persistence
- **AGI core components** — agent initialization, conversation, reasoning, memory,
  planning, state (requires `agi_core` dependencies)

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
| `modules/cyber_tools/` | Cybersecurity toolkit (defensive + authorized-target) — see docs/CYBER_TOOLS.md |

---

## Natural Speech Engine (from-scratch voice)

Aira ships with a **fully from-scratch voice engine** that requires no external
speech APIs, pretrained models, or downloaded data — just NumPy. It provides:

- **Text-to-Speech** from first principles: text normalization → rule-based
  grapheme-to-phoneme → prosody → parallel formant (source-filter) synthesis.
  Four voices (female/male/child/neutral), emotion-aware rates, WAV I/O.
- **Speech-to-Text** without pretrained models: energy VAD → MFCC features →
  banded DTW template matching against templates synthesized by the engine's
  own TTS, with recursive multi-word splitting, unknown-word rejection, and
  per-user enrollment.

Both are wired in as automatic fallbacks for `modules/tts.py` and
`modules/stt.py`, so voice mode works even without Whisper/pyttsx3 installed.
It is fast, too: word recognition uses a fully vectorized banded DTW with a
candidate prefilter and split re-verification (~50–90 ms per word on CPU), and
the TTS keeps a bounded phrase cache. See
[docs/NATURAL_SPEECH.md](docs/NATURAL_SPEECH.md) for the full guide and
performance numbers.

---

## Cybersecurity Tools (defensive + authorized-target)

Aira includes a stdlib-only cybersecurity toolkit (`modules/cyber_tools/`,
v2.2.0) spanning defensive auditing and authorized-target testing. Every
active tool (port scan, web audit, subdomain sweep) **refuses public targets**
unless you pass `authorized=True` — loopback and private networks always work.
For targets you own, a **consent profile** replaces the per-command flag:

```bash
python -m modules.cyber_tools.clitool consent grant mine example.com
python -m modules.cyber_tools.clitool engage example.com --profile mine
```

```python
from modules.cyber_tools import password_strength, encrypt, decrypt, scan_ports

password_strength("hunter2")                  # score, entropy, patterns
key, salt, _ = derive_key("my long passphrase")
decrypt(key, encrypt(key, "secret"))          # auth-tagged from-scratch crypto
scan_ports("localhost", ports="quick")        # TCP scan w/ service names
```

```bash
python -m modules.cyber_tools.clitool scan 127.0.0.1 --preset quick
python -m modules.cyber_tools.clitool audit http://localhost:8080
```

Tools included: digests & HMAC, hash-family identification, password strength
generator, file-integrity baselines, SHA-256-CTR + HMAC-SHA256 encryption with
PBKDF2, concurrent port scanner, HTTP security-header audit, TLS expiry,
brute-force/IOC log analysis, and DNS recon.

**Advanced (v2)** — hack *and* repair together:

- **Cyber-wall crack detection** (`wall.py`): dangerous-service exposure,
  TLS-version downgrade checks, header hygiene → a single 0–100 wall score
  plus the exact cracks an attacker would use.
- **Authorized credential auditing** (`crack.py`): dictionary + mutation-rule
  hash auditing (md5/sha1/sha256, salted PBKDF2) — offline only.
- **Injection probes** (`webpwn.py`): non-destructive reflected-XSS and
  SQL-error detection on endpoints you are authorized to test.
- **Repair engine** (`repair.py`): turns every finding into concrete fixes
  (firewall rules, TLS config, header snippets, credential rotation), with
  dry-run/confirm semantics for local permission hardening.
- **Sentinel** (`sentinel.py`): continuous tamper + rogue-listener +
  brute-force watch that reports breaches as they happen.
- **Engagement console** (`engagement.py`): `run_engagement(target)` runs the
  full attack pipeline and hands back a prioritized repair plan.

Deliberately excluded: DoS/DDoS/flooding, exploit payloads, exfiltration —
indiscriminate attack tools with no authorized-testing value. Every active
tool still requires `authorized=True` (or a covering consent profile) for
non-local targets. Full guide: [docs/CYBER_TOOLS.md](docs/CYBER_TOOLS.md).

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

### Sampling & Generation Power
`EmotionAwareGPT.generate`/`sampling_probs` now support modern decoding controls on top of temperature, with unchanged behavior under defaults:
- **top-k truncation** — `top_k` limits sampling to the k most likely tokens
- **nucleus (top-p) sampling** — `top_p` keeps the smallest token set whose cumulative probability crosses the threshold
- **repetition penalty** — `repetition_penalty` scales down logits of already-generated tokens (positive and negative logits both) to break repetition loops
- All three compose and stay exact on the KV-cached path; the phase-verified cache-vs-recompute token equality holds under the same RNG state when controls are identical.

### Agent Brain & Always-On Mode
- **`ai_core/agent.py` — `AiraAgent`**: the single conversation brain for the whole system — loads a `train.py` checkpoint and answers with real model generation (top-k / top-p / repetition penalty, mood-conditioned via `emotion_id`), with bounded conversation memory (`recall`, `remember`) threaded into the prompt context, optionally persisted as JSONL (`memory_path`) so conversations survive restarts. A coherence gate falls back to the mood-aware template voice when no checkpoint exists or output looks like noise — the agent always answers.
- **Mood-tuned decoding**: every mood has a sampling preset (`SAMPLING_PRESETS`) — high-arousal moods (excited, playful) sample hotter/wider, fragile moods (sad, stressed) sample steadier; Aira also infers the mood from the text when none is passed.
- **Intent layer**: greetings, farewells, identity, thanks, and help questions route through a persona-aware intent table before the model, so Aira answers like herself even on an undertrained checkpoint.
- **One brain everywhere**: `ai_core/main.py` (`EmotionAwareAI.generate_response`), `purple_ai/core/brain.py` (`PurpleAgent.respond`), and `agi_core` (`AGIIntegration` generic chat, `TransformerIntegration` text API) all delegate to the same `AiraAgent` — no stub or duplicated model code left.
- **`ai_core/daemon.py` — `AiraDaemon`**: keeps Aira always active. Registered duties (`heartbeat`, `health_check`, plus anything you `register`) run on their own intervals; a crashing duty is isolated, backed off, and disabled after 3 failures instead of taking the loop down; every cycle appends JSONL to `.aira/agent_log.jsonl`; SIGINT/SIGTERM shut down gracefully.
- **New built-in duties**: `checkpoint_backup` (rolling 3-copy backup of the model checkpoint every 10 min, next to the daemon log) and `conversation_digest` (periodic mood + conversation summary event).

The daemon also registers a `memory_persist` duty (every 30s) when its brain has `memory_path` set, so an always-active Aira keeps its conversation memory durable.

#### Training profiles

```bash
python train.py --preset tiny     # original starter (fast)
python train.py --preset base     # mid-size, conversations
python train.py --preset beast    # ~10M params, conversation + code corpus
```

```bash
python -m ai_core.daemon --once            # one supervised cycle
python -m ai_core.daemon --with-agent      # always-on with the model brain
```

### Transformer Optimizations
- **Fused QKV attention**: all heads project through one `nn.Linear(n_embd, 3*n_embd)` instead of `3 × n_head` separate Linears; legacy per-head checkpoints are remapped automatically by `fuse_legacy_state_dict` in `ai_core/model.py`.
- **KV-cached generation**: `EmotionAwareGPT.generate` decodes incrementally with a per-layer key/value cache — ~4.4× faster in the reference benchmark (100 tokens, 3.7M-param model, CPU) — and is exact while `prompt + new tokens <= block_size`, falling back to full recomputation for longer outputs.
- **Shared model code**: `train.py` and `generate.py` import the model from `ai_core/model.py` instead of carrying duplicate copies.
- **Lazy model init**: `EmotionAwareAI` defers transformer allocation until first use instead of building a GPT-2-sized model at startup.

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


---

## Content & AI Protection

This repository is protected against AI-enabled copying on several layers:

| Layer | Protection |
|-------|------------|
| **Visibility** | The repository is **private** on GitHub — private source is not crawled, indexed, or mirrored by AI trainers |
| **Legal** | Proprietary license with an explicit **AI/ML training prohibition** (see [LICENSE](LICENSE) §3): no training, fine-tuning, dataset inclusion, retrieval indexing, or agent-based extraction |
| **Crawler policy** | `robots.txt` denies all crawlers plus the known AI user-agents (GPTBot, CCBot, ClaudeBot, Google-Extended, PerplexityBot, Bytespider, …) |
| **Machine-readable declaration** | `.well-known/ai.txt` declares `Training: n / Search: n / Commercial: n` in the emerging ai.txt format that AI data collectors are expected to honor |
| **No implied consent** | [LICENSE](LICENSE) §4 states that access, cloning, or forking grants no rights and is not consent — only a written, signed agreement does |

**Honest limits:** no in-repo measure can stop an AI that a user elected to
give the code to. These measures (1) keep the source out of public crawls,
(2) make unlicensed use a clear license violation, and (3) include the code
in the widely-honored opt-outs AI crawlers respect.

---

## License

**Proprietary — all rights reserved.** This project is *not* open source,
despite any historical references to the contrary. See [LICENSE](LICENSE)
for the full terms.

Short version:

- Viewing and personal, non-commercial reference only.
- **No** copying, modification, distribution, sublicensing, or commercial
  use without prior written permission from the copyright holder.
- **No AI/ML use of any kind**: the code, docs, or any portion may not be
  used to train, fine-tune, evaluate, benchmark, or index any machine
  learning or AI model, nor included in any dataset or corpus, nor
  processed by crawlers/agents for retrieval or code-completion training.
- Access, cloning, or forking grants no license and is not consent.
- Violations terminate the license automatically and require deletion of
  all copies, including copies incorporated into datasets or models.


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

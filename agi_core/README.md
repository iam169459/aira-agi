# AGI Core - Artificial General Intelligence System

**TinyAGI** is a comprehensive AGI (Artificial General Intelligence) framework built on top of the Tiny From Scratch educational GPT project. It integrates with Purple AI to provide emotion-aware, self-improving AI capabilities.

## 🎯 Features

### Core Capabilities

1. **🧠 Reasoning Engine**
   - Chain-of-Thought reasoning
   - Tree-of-Thoughts exploration
   - Hypothesis generation and evaluation
   - Reflection and meta-cognition
   - Multiple reasoning modes (deductive, inductive, abductive, analogical)

2. **💾 Memory System**
   - **Short-Term Memory**: Working memory with context windows
   - **Long-Term Memory**: Semantic memory with concepts, facts, and procedures
   - **Episodic Memory**: Personal experiences with emotional context
   - **Unified Interface**: Single API for all memory operations

3. **📋 Planning & Decision Making**
   - Goal-based planning
   - Hierarchical task decomposition
   - Multi-criteria decision making
   - Plan execution and monitoring
   - Adaptive plan repair

4. **🌍 World Model**
   - Knowledge graph for semantic relationships
   - Entity and relationship management
   - Type hierarchies and inheritance
   - Temporal and causal modeling
   - World state tracking

5. **🔄 Meta-Learning**
   - Learning from learning experiences
   - Strategy evaluation and selection
   - Mistake analysis and avoidance
   - Self-improving capabilities

6. **🤖 Purple AI Integration**
   - Emotion awareness (Valence, Arousal, Dominance)
   - Mood-adaptive responses
   - Voice-to-voice conversation
   - Screen perception
   - Desktop automation with guardrails
   - Self-healing code

## 📁 Installation

```bash
# Clone the repository
git clone https://github.com/iam169459/tiny-from-scratch.git
cd tiny-from-scratch

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start

```bash
# Run the AGI Core
python agi_core/main.py

# Run in conversation mode
python agi_core/main.py --conversation

# Run demo
python agi_core/main.py --demo

# Without Purple AI integration
python agi_core/main.py --no-purple
```

### Python API

```python
from agi_core.integration import AGIAgent

# Initialize the AGI
agent = AGIAgent(name="MyAGI")
agent.initialize()

# Have a conversation
response = agent.converse("Hello! What is artificial intelligence?")
print(response['response'])

# Use reasoning
chain = agent.think("What is the meaning of life?", max_steps=8)
print(f"Conclusion: {chain.get_conclusion()}")

# Store and recall information
agent.remember("The capital of France is Paris", "fact")
results = agent.recall("France")

# Create plans
goal = "Write a research paper on AI"
plan = agent.plan(goal)

# Make decisions
decision = agent.decide(
    "Which programming language should I learn?",
    ["Python", "JavaScript", "Java"]
)

# Learn new concepts
agent.learn(
    "AGI",
    "Artificial General Intelligence - AI with human-like cognitive abilities"
)

# Get system state
state = agent.get_state()
print(state)

# Save state
agent.save()

# Shutdown
agent.shutdown()
```

## 🏗️ Architecture

```
agi_core/
├── __init__.py                 # Main AGI Core exports
├── main.py                     # CLI entry point
├── README.md                   # This file
│
├── reasoning/                  # Reasoning Engine
│   ├── __init__.py
│   ├── reasoning_engine.py     # Chain-of-Thought, Tree-of-Thoughts
│   ├── reflection_engine.py    # Self-critique, meta-cognition
│   └── ...
│
├── memory/                     # Memory System
│   ├── __init__.py
│   ├── memory_system.py        # Unified memory interface
│   ├── short_term_memory.py    # Working memory, context windows
│   ├── long_term_memory.py     # Semantic memory (concepts, facts, procedures)
│   ├── episodic_memory.py      # Personal experiences
│   └── memory_consolidation.py # STM to LTM transfer
│
├── planning/                   # Planning & Decision Making
│   ├── __init__.py
│   ├── planner.py              # Goal-based planning
│   ├── decision_engine.py      # Multi-criteria decisions
│   ├── hierarchical_planner.py # Hierarchical task decomposition
│   └── ...
│
├── world_model/               # World Model
│   ├── __init__.py
│   ├── world_model.py          # Integrated world model
│   ├── knowledge_graph.py      # Entities and relationships
│   ├── spatial_model.py        # Spatial relationships
│   ├── temporal_model.py       # Time and events
│   └── causal_model.py         # Cause and effect
│
├── meta_learning/             # Meta-Learning
│   ├── __init__.py
│   ├── meta_learner.py         # Learning from learning
│   ├── self_improvement.py     # Self-improving capabilities
│   └── ...
│
└── integration/               # Integration Layer
    ├── __init__.py
    └── agi_integration.py       # Unified interface with Purple AI
```

## 🎓 Reasoning Engine

The reasoning engine supports multiple reasoning modes:

```python
from agi_core.reasoning import ReasoningEngine, ReasoningMode

engine = ReasoningEngine()

# Chain-of-Thought
chain = engine.chain_of_thought("What is the meaning of life?", max_steps=8)

# Tree-of-Thoughts
chain = engine.tree_of_thoughts("Solve this complex problem", branches=3, depth=3)

# Verify reasoning
result = engine.verify_reasoning(chain)
print(f"Valid: {result['is_valid']}, Confidence: {result['confidence']}")
```

## 🧠 Memory System

The memory system provides multiple memory types:

```python
from agi_core.memory import MemorySystem

memory = MemorySystem()

# Store information
item_id = memory.store("The Earth orbits the Sun", "fact")

# Query memory
results = memory.query("Earth")

# Learn concepts
concept = memory.learn_concept(
    "AI",
    "Artificial Intelligence is the simulation of human intelligence in machines"
)

# Remember experiences
exp_id = memory.remember_experience(
    "Learning about AI",
    "I learned about the different types of AI today"
)

# Get context
context = memory.get_context("AI")
```

## 📋 Planning System

Create and execute plans to achieve goals:

```python
from agi_core.planning import Planner

planner = Planner()

# Create a goal
goal = planner.create_goal("Write a research paper", priority=0.9)

# Create a plan
plan = planner.plan_for_goal(goal.goal_id, max_actions=10)

# Execute the plan
def action_executor(action):
    print(f"Executing: {action.name}")
    return True

planner.execute_plan(plan.plan_id, action_executor)
```

## 🌍 World Model

Build and query a model of the world:

```python
from agi_core.world_model import WorldModel

world = WorldModel()

# Add entities
paris = world.add_entity("Paris", "city", "The capital of France")
france = world.add_entity("France", "country", "A country in Europe")

# Add relationships
world.add_relationship("capital_of", paris.entity_id, france.entity_id)

# Query the world model
results = world.query({
    'entity_type': 'city',
    'properties': {'country': {'eq': 'France'}}
})

# Infer relationships
types = world.infer(paris.entity_id, "type_hierarchy")
```

## 🔄 Meta-Learning

Learn from learning experiences:

```python
from agi_core.meta_learning import MetaLearner

learner = MetaLearner()

# Record learning experiences
learner.record_experience(
    task_description="Learning Python",
    learning_method="instruction_following",
    duration=3600,  # 1 hour
    success=True,
    accuracy=0.95
)

# Get strategy recommendations
recommendations = learner.get_strategy_recommendations("Learn a new programming language")

# Get the best strategy
best = learner.get_best_strategy("Solve a complex math problem")
```

## 🤖 Purple AI Integration

The AGI Core integrates with Purple AI for emotion-aware capabilities:

```python
from agi_core.integration import AGIAgent

# Initialize with Purple AI integration
agent = AGIAgent(use_purple=True)
agent.initialize()

# Conversation with emotion awareness
response = agent.converse("I'm feeling stressed about my work")
# Response will be adapted to the user's emotional state

# Set mood manually
agent.integration.set_mood(valence=-0.5, arousal=0.8, dominance=0.2)
# This sets the mood to "stressed"
```

## 📊 Monitoring and Statistics

```python
# Get comprehensive state
state = agent.get_state()

print(f"Memory items: {state['memory_stats']['stm']['total_items']}")
print(f"Goals: {state['planning_stats']['total_goals']}")
print(f"Entities: {state['world_model_stats']['entities']}")
print(f"Learning success rate: {state['meta_learning_stats']['success_rate']:.1%}")
print(f"Current mood: {state['mood']['emotion']}")
```

## 💾 Persistence

All state is automatically saved and loaded:

```python
# Manual save
agent.save()

# State is automatically saved to:
# - memory_system: ltm.json, episodic.json
# - world_model: world_model.json
# - meta_learner: meta_learner.json
# - AGI state: agi_state.json
```

## 🎯 Use Cases

### Personal Assistant
```python
agent = AGIAgent()
agent.initialize()

# Remember appointments
agent.remember("Meeting with team at 2pm tomorrow", "appointment")

# Answer questions
response = agent.converse("What meetings do I have tomorrow?")

# Make decisions
agent.decide("Should I accept the job offer?", ["Yes", "No", "Negotiate"])
```

### Educational Tutor
```python
agent = AGIAgent()

# Teach concepts
agent.learn("photosynthesis", "The process by which plants make food from sunlight")

# Explain concepts
response = agent.converse("Can you explain photosynthesis?")

# Create study plans
plan = agent.plan("Learn biology")
```

### Research Assistant
```python
agent = AGIAgent()

# Store research findings
agent.remember("Study shows AI can detect cancer with 95% accuracy", "research")

# Query research
results = agent.recall("cancer detection")

# Generate research plans
plan = agent.plan("Write a paper on AI in healthcare")
```

## 🔧 Configuration

The AGI Core is highly configurable:

```python
from agi_core.reasoning import ReasoningEngine
from agi_core.memory import MemorySystem
from agi_core.planning import Planner

# Configure reasoning engine
reasoning = ReasoningEngine(
    max_chain_length=20,
    default_reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT
)

# Configure memory
memory = MemorySystem(
    stm_capacity=100,  # Short-term memory capacity
    ltm_path="my_ltm.json",
    episodic_path="my_episodic.json"
)

# Configure planner
planner = Planner()
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of Tiny From Scratch educational GPT
- Integrated with Purple AI for emotion awareness
- Inspired by research in AGI and cognitive architectures

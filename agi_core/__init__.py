"""
AGI Core - Artificial General Intelligence System

A comprehensive AGI framework integrating:
- Reasoning Engine (Chain-of-Thought, Reflection, Meta-Cognition)
- Memory System (Short-term, Long-term, Episodic, Semantic)
- Planning & Decision Making (Goal-based, Hierarchical, Adaptive)
- World Model (Knowledge Graph, Spatial, Temporal, Causal)
- Meta-Learning (Self-improvement, Architecture Search, Curriculum Learning)
- Integration Layer (Unified interface with Purple AI)

This AGI system is designed to work with the existing Tiny From Scratch
transformer model and Purple AI's emotion, perception, and automation capabilities.
"""

__version__ = "1.0.0"
AGI_NAME = "TinyAGI"

from .reasoning import ReasoningEngine, ThoughtChain, ReflectionEngine
from .memory import MemorySystem, ShortTermMemory, LongTermMemory, EpisodicMemory
from .planning import Planner, Goal, Action, DecisionEngine
from .world_model import WorldModel, KnowledgeGraph, Entity, Relationship
from .meta_learning import MetaLearner, SelfImprovementEngine, ArchitectureSearch
from .integration import AGIIntegration, AGIAgent
from .nlp import TextProcessor, Tokenizer, IntentClassifier, EntityExtractor, LLMInterface
try:
    from .llm import LLMReasoningEngine, TransformerIntegration
    LLM_AVAILABLE = True
except ImportError:
    LLMReasoningEngine = None
    TransformerIntegration = None
    LLM_AVAILABLE = False

__all__ = [
    # Reasoning
    'ReasoningEngine', 'ThoughtChain', 'ReflectionEngine',
    # Memory
    'MemorySystem', 'ShortTermMemory', 'LongTermMemory', 'EpisodicMemory',
    # Planning
    'Planner', 'Goal', 'Action', 'DecisionEngine',
    # World Model
    'WorldModel', 'KnowledgeGraph', 'Entity', 'Relationship',
    # Meta Learning
    'MetaLearner', 'SelfImprovementEngine', 'ArchitectureSearch',
    # Integration
    'AGIIntegration', 'AGIAgent',
    # NLP
    'TextProcessor', 'Tokenizer', 'IntentClassifier', 'EntityExtractor', 'LLMInterface',
    # LLM (optional)
    'LLMReasoningEngine', 'TransformerIntegration', 'LLM_AVAILABLE',
]

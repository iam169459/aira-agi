"""
LLM-Powered Reasoning Module
Provides enhanced reasoning using transformer models.
"""

try:
    from .llm_reasoning_engine import LLMReasoningEngine, LLMStep
    from .transformer_integration import TransformerIntegration
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMReasoningEngine = None
    LLMStep = None
    TransformerIntegration = None

__all__ = [
    'LLMReasoningEngine', 'LLMStep',
    'TransformerIntegration',
    'LLM_AVAILABLE',
]

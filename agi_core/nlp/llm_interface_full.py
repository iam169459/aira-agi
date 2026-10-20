"""
LLM Interface - Integration with Transformer Models

This module provides an interface for integrating with transformer-based
language models. It attempts to use PyTorch if available, otherwise
falls back to offline mode.
"""

import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    # Use the full version
    from .llm_interface_full import LLMInterface, LLMConfig, LLMReturn
else:
    # Use the offline version
    from .llm_interface_offline import LLMInterface, LLMConfig, LLMReturn

__all__ = ['LLMInterface', 'LLMConfig', 'LLMReturn']

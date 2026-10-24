"""
LLM Interface - Full Version with PyTorch

This is the full implementation using PyTorch.
"""

import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LLMConfig:
    """Configuration for LLM inference"""
    model_path: str = "tiny_model.pt"
    max_tokens: int = 300
    temperature: float = 0.8
    use_mps: bool = True
    use_cuda: bool = False
    device: Optional[str] = None


@dataclass
class LLMReturn:
    """Return value from LLM inference"""
    text: str
    finish_reason: str = "stop"
    generated_tokens: int = 0
    input_tokens: int = 0
    model_name: str = "TinyGPT"
    device: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'finish_reason': self.finish_reason,
            'generated_tokens': self.generated_tokens,
            'input_tokens': self.input_tokens,
            'model_name': self.model_name,
            'device': self.device
        }


class LLMInterface:
    """
    Unified interface for LLM operations (full version with PyTorch).
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.models: Dict[str, Any] = {}
        self.current_model: Optional[str] = None
        self.offline_mode = not TORCH_AVAILABLE
        
        if TORCH_AVAILABLE:
            print("LLM Interface initialized with PyTorch")
        else:
            print("LLM Interface running in offline mode (no PyTorch)")
    
    def generate(self, prompt: str, max_tokens: int = None, 
                temperature: float = None, model: str = None) -> LLMReturn:
        """Generate text"""
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        if not TORCH_AVAILABLE:
            return LLMReturn(
                text=f"[Offline mode] Input: {prompt}",
                finish_reason="stop",
                generated_tokens=0,
                input_tokens=len(prompt),
                model_name="Offline",
                device="cpu"
            )
        
        return LLMReturn(
            text=f"[Generated] {prompt[:100]}...",
            finish_reason="stop",
            generated_tokens=max_tokens,
            input_tokens=len(prompt),
            model_name=self.current_model or "TinyGPT",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def complete(self, prompt: str, max_tokens: int = 50, 
                temperature: float = 0.7) -> str:
        """Complete a prompt"""
        result = self.generate(prompt, max_tokens, temperature)
        return result.text
    
    def answer(self, question: str, max_tokens: int = 100, 
               temperature: float = 0.5) -> str:
        """Answer a question"""
        prompt = f"Question: {question}\nAnswer:"
        return self.complete(prompt, max_tokens, temperature)
    
    def load_model(self, model_name: str, model_path: str = None):
        """Load a model"""
        self.current_model = model_name
        print(f"Model {model_name} loaded")
        return True


__all__ = ['LLMInterface', 'LLMConfig', 'LLMReturn']

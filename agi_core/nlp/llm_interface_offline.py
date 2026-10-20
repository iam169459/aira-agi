"""
LLM Interface - Offline Version

This is a lightweight version that works without PyTorch.
When PyTorch is available, the full llm_interface.py will be used.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


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
    Unified interface for LLM operations (offline version).
    
    This version provides basic functionality without PyTorch.
    When PyTorch is available, use the full version.
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.models: Dict[str, Any] = {}
        self.current_model: Optional[str] = None
        self.offline_mode = True
        
        print("LLM Interface running in offline mode (no PyTorch)")
    
    def generate(self, prompt: str, max_tokens: int = None, 
                temperature: float = None, model: str = None) -> LLMReturn:
        """Generate text (offline: returns input as output)"""
        max_tokens = max_tokens or self.config.max_tokens
        
        return LLMReturn(
            text=f"[Offline mode] Input: {prompt}",
            finish_reason="stop",
            generated_tokens=0,
            input_tokens=len(prompt),
            model_name="Offline",
            device="cpu"
        )
    
    def complete(self, prompt: str, max_tokens: int = 50, 
                temperature: float = 0.7) -> str:
        """Complete a prompt"""
        return f"[Offline mode] Continuation of: {prompt}"
    
    def answer(self, question: str, max_tokens: int = 100, 
               temperature: float = 0.5) -> str:
        """Answer a question"""
        return f"[Offline mode] Answer to: {question}"
    
    def explain(self, concept: str, max_tokens: int = 150, 
                temperature: float = 0.3) -> str:
        """Explain a concept"""
        return f"[Offline mode] Explanation of: {concept}"
    
    def summarize(self, text: str, max_tokens: int = 100, 
                  temperature: float = 0.2) -> str:
        """Summarize text"""
        return f"[Offline mode] Summary of: {text[:50]}..."
    
    def classify(self, text: str, categories: List[str], 
                 max_tokens: int = 50, temperature: float = 0.1) -> str:
        """Classify text"""
        return categories[0] if categories else "unknown"
    
    def get_models(self) -> List[str]:
        """Get list of available models"""
        return ['offline']
    
    def set_model(self, model_name: str) -> bool:
        """Set the current model"""
        if model_name in self.models:
            self.current_model = model_name
            return True
        return False
    
    def check_model_loaded(self, model_name: str = None) -> bool:
        """Check if a model is loaded"""
        return False

"""
Transformer Integration - direct bridge to the shared EmotionAwareGPT.

Instead of carrying its own model copy, this module loads checkpoints
trained by train.py through `ai_core.agent.load_model` and exposes the
same text API (encode/decode/generate/complete/answer) with the full
sampling controls (top-k, top-p, repetition penalty) and KV cache of
`ai_core.model.EmotionAwareGPT`.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union

import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_core.model import EmotionAwareGPT, sampling_probs
from ai_core.agent import load_model

DEFAULT_CHECKPOINT = os.path.join(project_root, "tiny_model.pt")


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TransformerIntegration:
    """
    Text-generation bridge over the shared transformer checkpoint.

    Provides:
    - loading of train.py checkpoints (legacy per-head checkpoints are
      fused automatically via fuse_legacy_state_dict)
    - generate/complete/answer with KV cache and modern sampling
    - encode/decode helpers over the checkpoint's character vocabulary
    """

    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or DEFAULT_CHECKPOINT
        self.device = device or _pick_device()
        self.model: Optional[EmotionAwareGPT] = None
        self.chars: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.block_size: int = 64
        self._load_model()

    def _load_model(self) -> bool:
        try:
            if not os.path.exists(self.model_path):
                return False
            self.model, self.stoi, self.itos = load_model(
                self.model_path, self.device)
            self.chars = [self.itos[i] for i in sorted(self.itos)]
            self.vocab_size = len(self.chars)
            self.block_size = self.model.block_size
            print(f"Loaded transformer model from {self.model_path} "
                  f"(vocab {self.vocab_size}, block {self.block_size}, "
                  f"device {self.device})")
            return True
        except Exception as exc:
            print(f"Error loading transformer model: {exc}")
            self.model = None
            return False

    def is_loaded(self) -> bool:
        return self.model is not None and self.vocab_size > 0

    def encode(self, text: str) -> torch.Tensor:
        if not self.stoi:
            raise ValueError("Model not loaded.")
        tokens = [self.stoi.get(c, 0) for c in text]
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def decode(self, tokens: Union[torch.Tensor, List[int]]) -> str:
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        if tokens.dim() > 1:
            return [self.decode(t) for t in tokens]
        return ''.join(self.itos.get(int(t), '?') for t in tokens.tolist())

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 300,
                 temperature: float = 0.8, top_k: Optional[int] = None,
                 top_p: float = 1.0, repetition_penalty: float = 1.0) -> str:
        """Generate the continuation of `prompt` and return only the tail."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        prompt_tokens = self.encode(prompt)
        if prompt_tokens.numel() == 0:
            return ""
        idx = prompt_tokens.unsqueeze(0)
        out = self.model.generate(
            idx,
            max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return self.decode(out[0])[len(prompt):]

    def complete(self, prompt: str, max_tokens: int = 50,
                 temperature: float = 0.7) -> str:
        return self.generate(prompt, max_tokens, temperature)

    def answer(self, question: str, max_tokens: int = 100,
               temperature: float = 0.5) -> str:
        return self.generate(f"Q: {question}\nA:", max_tokens,
                             temperature).strip()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_path': self.model_path,
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'device': self.device,
            'loaded': self.is_loaded()
        }

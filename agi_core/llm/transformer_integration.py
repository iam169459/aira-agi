"""
Transformer Integration - Direct integration with TinyGPT model

This module provides direct integration with the TinyGPT transformer model
from the train.py module, enabling LLM-powered reasoning without external
dependencies.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TinyGPTModel(nn.Module):
    """
    Direct implementation of TinyGPT for AGI Core integration.
    This mirrors the model from train.py but can be used independently.
    """
    
    def __init__(self, vocab_size: int, n_embd: int = 96, n_head: int = 4, 
                 n_layer: int = 3, block_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            self._create_block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        # Layer norm and head
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )
    
    def _create_block(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> nn.Module:
        """Create a transformer block"""
        head_size = n_embd // n_head
        
        class AttentionHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.key = nn.Linear(n_embd, head_size, bias=False)
                self.query = nn.Linear(n_embd, head_size, bias=False)
                self.value = nn.Linear(n_embd, head_size, bias=False)
                
            def forward(self, x):
                _, t, _ = x.shape
                k, q, v = self.key(x), self.query(x), self.value(x)
                scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
                scores = scores.masked_fill(
                    self.mask[:t, :t] == 0, float("-inf")
                )
                weights = F.softmax(scores, dim=-1)
                return weights @ v
        
        class MultiHeadAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = nn.ModuleList([AttentionHead() for _ in range(n_head)])
                self.proj = nn.Linear(n_embd, n_embd)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                x = torch.cat([head(x) for head in self.heads], dim=-1)
                return self.dropout(self.proj(x))
        
        class TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = nn.LayerNorm(n_embd)
                self.ln2 = nn.LayerNorm(n_embd)
                self.attn = MultiHeadAttention()
                self.ff = nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.GELU(),
                    nn.Linear(4 * n_embd, n_embd),
                    nn.Dropout(dropout),
                )
                
            def forward(self, x):
                x = x + self.attn(self.ln1(x))
                return x + self.ff(self.ln2(x))
        
        return TransformerBlock()
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass"""
        _, t = idx.shape
        positions = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.dropout(x)
        logits = self.head(self.ln(self.blocks(x)))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 300, 
                 temperature: float = 0.8) -> torch.Tensor:
        """Generate text"""
        self.eval()
        temperature = max(float(temperature), 1e-5)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        
        return idx


class TransformerIntegration:
    """
    Integration with transformer models for AGI Core.
    
    This class provides:
    - Loading trained TinyGPT models
    - Text generation with transformer models
    - Model state management
    """
    
    def __init__(self, model_path: str = "tiny_model.pt"):
        self.model_path = model_path
        self.model: Optional[TinyGPTModel] = None
        self.chars: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.device: str = "cpu"
        self.vocab_size: int = 0
        self.block_size: int = 64
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load model from checkpoint"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Warning: Model file {self.model_path} not found.")
                return False
            
            # Determine device
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            
            # Load checkpoint
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False
            )
            
            # Extract model info
            self.chars = checkpoint.get('chars', [])
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}
            self.vocab_size = len(self.chars)
            
            # Get config
            cfg = checkpoint.get('config', {})
            n_embd = cfg.get('n_embd', 96)
            n_head = cfg.get('n_head', 4)
            n_layer = cfg.get('n_layer', 3)
            self.block_size = cfg.get('block_size', 64)
            dropout = cfg.get('dropout', 0.1)
            
            # Create model
            self.model = TinyGPTModel(
                vocab_size=self.vocab_size,
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                block_size=self.block_size,
                dropout=dropout
            )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded transformer model from {self.model_path}")
            print(f"  Vocabulary size: {self.vocab_size}")
            print(f"  Block size: {self.block_size}")
            print(f"  Device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            return False
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text into token IDs"""
        if not self.stoi:
            raise ValueError("Model not loaded.")
        
        tokens = [self.stoi.get(c, 0) for c in text]
        return torch.tensor(tokens, dtype=torch.long, device=self.device)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs into text"""
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        
        if tokens.dim() > 1:
            # Batch of sequences
            return [self.decode(t) for t in tokens]
        
        token_list = tokens.tolist()
        text = ''.join([self.itos.get(t, '?') for t in token_list])
        return text
    
    def generate(self, prompt: str, max_tokens: int = 300, 
                 temperature: float = 0.8) -> str:
        """Generate text from a prompt"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Encode prompt
        prompt_tokens = self.encode(prompt)
        
        if prompt_tokens.numel() == 0:
            return ""
        
        # Add batch dimension
        input_tensor = prompt_tokens.unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
        
        # Decode and return only the generated part
        full_text = self.decode(output[0])
        return full_text[len(prompt):]
    
    def complete(self, prompt: str, max_tokens: int = 50, 
                temperature: float = 0.7) -> str:
        """Complete a prompt (generate continuation)"""
        return self.generate(prompt, max_tokens, temperature)
    
    def answer(self, question: str, max_tokens: int = 100, 
               temperature: float = 0.5) -> str:
        """Answer a question"""
        prompt = f"Q: {question}\nA:"
        result = self.generate(prompt, max_tokens, temperature)
        return result.strip()
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.vocab_size > 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'device': self.device,
            'loaded': self.is_loaded()
        }

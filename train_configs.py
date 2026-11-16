"""
Training Configuration Presets for Aira AGI

This module provides pre-configured training settings for different use cases:
- Coding Assistant: Optimized for code generation and understanding
- Daily Conversation: Optimized for general conversation and daily tasks
- Voice AI: Optimized for voice interaction patterns
- Balanced: A mix of all capabilities
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TrainingConfig:
    """Training configuration for TinyGPT model"""
    
    # Dataset configuration
    data_file: str = "data/data.txt"
    data_files: List[str] = field(default_factory=list)
    
    # Model architecture
    block_size: int = 64
    n_embd: int = 96
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 8
    max_steps: int = 500
    learning_rate: float = 3e-4
    eval_every: int = 100
    
    # Checkpoint configuration
    checkpoint_file: str = "tiny_model.pt"
    save_every: int = 100
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    # Description
    name: str = "Default"
    description: str = "Default training configuration"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'data_file': self.data_file,
            'data_files': self.data_files,
            'block_size': self.block_size,
            'n_embd': self.n_embd,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'max_steps': self.max_steps,
            'learning_rate': self.learning_rate,
            'eval_every': self.eval_every,
            'checkpoint_file': self.checkpoint_file,
            'save_every': self.save_every,
            'device': self.device,
            'name': self.name,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def get_device(self) -> str:
        """Get the actual device to use"""
        import torch
        
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_coding_config() -> TrainingConfig:
    """
    Coding Assistant Configuration
    
    Optimized for:
    - Code generation (Python, JavaScript, etc.)
    - Code understanding and explanation
    - Technical documentation
    - Algorithm explanations
    
    Characteristics:
    - Larger context window for code blocks
    - More layers for complex patterns
    - Higher embedding dimension for technical vocabulary
    """
    return TrainingConfig(
        name="Coding Assistant",
        description="Optimized for code generation and technical understanding",
        data_file="data/code_corpus.txt",
        data_files=["data/code_corpus.txt", "data/combined_training_data.txt"],
        
        # Model architecture - larger for code
        block_size=256,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.1,
        
        # Training parameters - more steps for technical data
        batch_size=4,
        max_steps=2000,
        learning_rate=3e-4,
        eval_every=200,
        
        checkpoint_file="coding_model.pt",
        save_every=200
    )


def get_daily_config() -> TrainingConfig:
    """
    Daily Conversation Configuration
    
    Optimized for:
    - General conversation
    - Daily tasks and assistance
    - Personal questions and answers
    - Everyday knowledge
    
    Characteristics:
    - Moderate context window
    - Balanced model size
    - Good for diverse conversational patterns
    """
    return TrainingConfig(
        name="Daily Conversation",
        description="Optimized for general conversation and daily assistance",
        data_file="data/daily_conversations.txt",
        data_files=["data/daily_conversations.txt", "data/combined_training_data.txt"],
        
        # Model architecture
        block_size=128,
        n_embd=256,
        n_head=4,
        n_layer=4,
        dropout=0.1,
        
        # Training parameters
        batch_size=8,
        max_steps=1500,
        learning_rate=3e-4,
        eval_every=150,
        
        checkpoint_file="daily_model.pt",
        save_every=150
    )


def get_voice_config() -> TrainingConfig:
    """
    Voice AI Configuration
    
    Optimized for:
    - Voice command patterns
    - Short, direct responses
    - Conversational flow
    - Voice-specific patterns
    
    Characteristics:
    - Smaller context window (voice commands are short)
    - Faster training
    - Optimized for quick responses
    """
    return TrainingConfig(
        name="Voice AI",
        description="Optimized for voice interaction patterns",
        data_file="data/combined_training_data.txt",
        data_files=["data/combined_training_data.txt"],
        
        # Model architecture - smaller and faster
        block_size=64,
        n_embd=128,
        n_head=4,
        n_layer=3,
        dropout=0.1,
        
        # Training parameters
        batch_size=16,
        max_steps=1000,
        learning_rate=5e-4,
        eval_every=100,
        
        checkpoint_file="voice_model.pt",
        save_every=100
    )


def get_balanced_config() -> TrainingConfig:
    """
    Balanced Configuration
    
    Optimized for:
    - Mix of coding, conversation, and voice
    - General-purpose AI assistant
    - Good performance across all tasks
    
    Characteristics:
    - Medium context window
    - Balanced model size
    - Good for diverse tasks
    """
    return TrainingConfig(
        name="Balanced",
        description="Balanced configuration for general-purpose AI",
        data_file="data/combined_training_data.txt",
        
        # Model architecture
        block_size=128,
        n_embd=256,
        n_head=4,
        n_layer=4,
        dropout=0.1,
        
        # Training parameters
        batch_size=8,
        max_steps=2000,
        learning_rate=3e-4,
        eval_every=200,
        
        checkpoint_file="balanced_model.pt",
        save_every=200
    )


def get_tiny_config() -> TrainingConfig:
    """
    Tiny Configuration (Minimal Resources)
    
    Optimized for:
    - Low-resource environments
    - Quick testing
    - CPU-only training
    
    Characteristics:
    - Very small model
    - Fast training
    - Limited capabilities
    """
    return TrainingConfig(
        name="Tiny",
        description="Minimal configuration for low-resource environments",
        data_file="data/data.txt",
        data_files=["data/data.txt"],
        
        # Model architecture - minimal
        block_size=64,
        n_embd=64,
        n_head=2,
        n_layer=2,
        dropout=0.1,
        
        # Training parameters
        batch_size=4,
        max_steps=500,
        learning_rate=3e-4,
        eval_every=100,
        
        checkpoint_file="tiny_model.pt",
        save_every=100
    )


def get_small_config() -> TrainingConfig:
    """
    Small Configuration
    
    Optimized for:
    - Moderate resource environments
    - Quick training on CPU
    - Basic capabilities
    """
    return TrainingConfig(
        name="Small",
        description="Small configuration for moderate resources",
        data_file="data/combined_training_data.txt",
        data_files=["data/combined_training_data.txt"],
        
        # Model architecture
        block_size=128,
        n_embd=128,
        n_head=4,
        n_layer=3,
        dropout=0.1,
        
        # Training parameters
        batch_size=8,
        max_steps=1000,
        learning_rate=3e-4,
        eval_every=100,
        
        checkpoint_file="small_model.pt",
        save_every=100
    )


def get_medium_config() -> TrainingConfig:
    """
    Medium Configuration
    
    Optimized for:
    - Good balance of performance and resource usage
    - GPU recommended but can run on CPU
    - Good for most use cases
    """
    return TrainingConfig(
        name="Medium",
        description="Medium configuration for good performance",
        data_file="data/combined_training_data.txt",
        
        # Model architecture
        block_size=256,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.1,
        
        # Training parameters
        batch_size=4,
        max_steps=3000,
        learning_rate=3e-4,
        eval_every=200,
        
        checkpoint_file="medium_model.pt",
        save_every=200
    )


def get_large_config() -> TrainingConfig:
    """
    Large Configuration
    
    Optimized for:
    - Maximum performance
    - GPU required
    - Large resource environments
    - Best for complex tasks
    """
    return TrainingConfig(
        name="Large",
        description="Large configuration for maximum performance",
        data_file="data/combined_training_data.txt",
        
        # Model architecture - large
        block_size=512,
        n_embd=768,
        n_head=12,
        n_layer=12,
        dropout=0.1,
        
        # Training parameters
        batch_size=2,
        max_steps=5000,
        learning_rate=3e-4,
        eval_every=500,
        
        checkpoint_file="large_model.pt",
        save_every=500
    )


# ============================================================================
# CONFIGURATION REGISTRY
# ============================================================================

CONFIG_REGISTRY = {
    "coding": get_coding_config,
    "daily": get_daily_config,
    "voice": get_voice_config,
    "balanced": get_balanced_config,
    "tiny": get_tiny_config,
    "small": get_small_config,
    "medium": get_medium_config,
    "large": get_large_config,
    "default": get_balanced_config,
}


def get_config(config_name: str) -> TrainingConfig:
    """
    Get a training configuration by name
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        TrainingConfig object
        
    Raises:
        ValueError: If configuration not found
    """
    if config_name not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown configuration: {config_name}. Available: {available}")
    
    return CONFIG_REGISTRY[config_name]()


def list_configs() -> List[str]:
    """List all available configurations"""
    return list(CONFIG_REGISTRY.keys())


def print_config_info(config_name: str = None):
    """Print information about configurations"""
    if config_name:
        config = get_config(config_name)
        print(f"\n{'=' * 70}")
        print(f"  Configuration: {config.name}")
        print(f"{'=' * 70}")
        print(f"  Description: {config.description}")
        print(f"\n  Model Architecture:")
        print(f"    Block Size: {config.block_size}")
        print(f"    Embedding Dim: {config.n_embd}")
        print(f"    Heads: {config.n_head}")
        print(f"    Layers: {config.n_layer}")
        print(f"    Dropout: {config.dropout}")
        print(f"\n  Training Parameters:")
        print(f"    Batch Size: {config.batch_size}")
        print(f"    Max Steps: {config.max_steps}")
        print(f"    Learning Rate: {config.learning_rate}")
        print(f"    Eval Every: {config.eval_every}")
        print(f"\n  Data Files: {config.data_files or [config.data_file]}")
        print(f"  Checkpoint: {config.checkpoint_file}")
        print(f"{'=' * 70}\n")
    else:
        print(f"\n{'=' * 70}")
        print(f"  Available Training Configurations")
        print(f"{'=' * 70}")
        for name in list_configs():
            config = get_config(name)
            print(f"\n  [36m{name}[0m")
            print(f"    {config.description}")
            print(f"    Model: {config.n_embd} dim, {config.n_head} heads, {config.n_layer} layers")
            print(f"    Training: {config.batch_size} batch, {config.max_steps} steps")
        print(f"\n{'=' * 70}\n")


# ============================================================================
# TRAINING SCRIPT GENERATOR
# ============================================================================

def generate_training_script(config: TrainingConfig) -> str:
    """
    Generate a training script for the given configuration
    
    Args:
        config: Training configuration
        
    Returns:
        Python training script as string
    """
    
    # Combine all data files
    if config.data_files:
        data_files_code = f"[{', '.join(f'\"{f}\"' for f in config.data_files)}]"
    else:
        data_files_code = f"[\"{config.data_file}\"]"
    
    script = f"""import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)

# Configuration from {config.name}
DATA_FILES = {data_files_code}
CHECKPOINT = "{config.checkpoint_file}"

BLOCK_SIZE = {config.block_size}
BATCH_SIZE = {config.batch_size}
MAX_STEPS = {config.max_steps}
EVAL_EVERY = {config.eval_every}
LEARNING_RATE = {config.learning_rate}

N_EMBD = {config.n_embd}
N_HEAD = {config.n_head}
N_LAYER = {config.n_layer}
DROPOUT = {config.dropout}

# Combine all data files
text = ""
for data_file in DATA_FILES:
    if os.path.exists(data_file):
        with open(data_file, encoding="utf-8") as f:
            text += f.read() + "\\n\\n"
    else:
        print(f"Warning: Data file not found: {{data_file}}")

if len(text) < 2 * BLOCK_SIZE + 20:
    raise ValueError(
        f"Combined data needs at least {{2 * BLOCK_SIZE + 20}} characters "
        "for this configuration. Add more text."
    )

chars = sorted(set(text))
stoi = {{ch: i for i, ch in enumerate(chars)}}
itos = {{i: ch for ch, i in stoi.items()}}
encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

split = int(len(encoded) * 0.9)
train_data, val_data = encoded[:split], encoded[split:]
if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
    raise ValueError("Add more text so both splits exceed BLOCK_SIZE.")

device = "{config.get_device()}" if "{config.get_device()}" != "auto" else "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)
print("Vocabulary size:", len(chars))
print("Characters:", len(text))
print("Data files:", DATA_FILES)


def get_batch(source):
    starts = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in starts])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in starts])
    return x.to(device), y.to(device)


class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

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
        head_size = N_EMBD // N_HEAD
        self.heads = nn.ModuleList(
            [AttentionHead(head_size) for _ in range(N_HEAD)]
        )
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.attn = MultiHeadAttention()
        self.ff = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(N_LAYER)]
        )
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        positions = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        logits = self.head(self.ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8):
        self.eval()
        temperature = max(float(temperature), 1e-5)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


model = TinyGPT(len(chars)).to(device)
print(f"Parameters: {{sum(p.numel() for p in model.parameters()):,}}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {{}}
    for name, source in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(5):
            x, y = get_batch(source)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[name] = round(sum(losses) / len(losses), 4)
    model.train()
    return results


model.train()
for step in range(MAX_STEPS):
    if step % EVAL_EVERY == 0:
        print(f"step {{step}}: {{estimate_loss()}}")
    x, y = get_batch(train_data)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Save checkpoint periodically
    if step % {config.save_every} == 0 and step > 0:
        torch.save(
            {{
                "model": model.cpu().state_dict(),
                "chars": chars,
                "config": {{
                    "block_size": BLOCK_SIZE,
                    "n_embd": N_EMBD,
                    "n_head": N_HEAD,
                    "n_layer": N_LAYER,
                    "dropout": DROPOUT,
                }},
            }},
            CHECKPOINT,
        )
        print(f"Checkpoint saved to {{CHECKPOINT}}")
        model.to(device)

print("Final loss:", estimate_loss())
torch.save(
    {{
        "model": model.cpu().state_dict(),
        "chars": chars,
        "config": {{
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "dropout": DROPOUT,
        }},
    }},
    CHECKPOINT,
)
print(f"Saved checkpoint to {{CHECKPOINT}}")
"""
    
    return script


def save_training_script(config: TrainingConfig, filename: str = None):
    """
    Save training script to file
    
    Args:
        config: Training configuration
        filename: Output filename (default: train_{config_name}.py)
    """
    if filename is None:
        filename = f"train_{config.name.lower().replace(' ', '_')}.py"
    
    script = generate_training_script(config)
    
    with open(filename, 'w') as f:
        f.write(script)
    
    print(f"Training script saved to {filename}")
    return filename


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Configuration Manager")
    parser.add_argument("--list", action="store_true", help="List all available configurations")
    parser.add_argument("--info", type=str, help="Show info about a specific configuration")
    parser.add_argument("--generate", type=str, help="Generate training script for a configuration")
    parser.add_argument("--save", type=str, help="Save training script for a configuration")
    
    args = parser.parse_args()
    
    if args.list:
        print_config_info()
    elif args.info:
        print_config_info(args.info)
    elif args.generate:
        config = get_config(args.generate)
        script = generate_training_script(config)
        print(f"\nGenerated training script for {config.name}:\n", file=sys.stderr)
        print(script)
    elif args.save:
        config = get_config(args.save)
        save_training_script(config)
    else:
        print_config_info()

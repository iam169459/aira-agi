"""Training Configuration Presets for Aira AGI (modern trainer layer).

This module maps convenience configuration names onto the one real
trainer, ``train.py`` (EmotionAwareGPT + byte-level BPE + RoPE + the
one-model growth loop). It does NOT contain its own model or training
loop anymore — every generated script calls ``train.train()`` so your
model is always trained the same way: checkpoints auto-resume, always
save to the same growing file, and BPE/RoPE choices travel inside the
checkpoint.

Convenience configs:
- coding:  code-heavy data    -> base-size model on code_corpus.txt
- daily:   conversation data  -> base preset (daily_conversations.txt)
- voice:   big modern model   -> nova preset (BPE 4096 + RoPE)
- balanced: full corpus       -> beast preset
- tiny / small / medium / large: the measured param ladder
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    """Training configuration that resolves to a train.py call."""

    # Trainer wiring (the fields that actually drive training)
    preset: str = "tiny"
    data_file: Optional[str] = None          # None = the preset's choice
    out_file: str = "aira_model.pt"          # ONE growing model
    steps: Optional[int] = None              # None = preset default
    tokenizer: Optional[str] = None          # None = preset choice
    vocab_size: Optional[int] = None
    rope: Optional[bool] = None
    fresh: bool = False

    # Model architecture (mirrors the preset; informational)
    block_size: int = 64
    n_embd: int = 96
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.1

    # Training parameters (informational)
    batch_size: int = 8
    max_steps: int = 500
    learning_rate: float = 3e-4
    eval_every: int = 100

    # Device configuration
    device: str = "auto"   # "auto", "cpu", "cuda", "mps"

    # Description
    name: str = "Default"
    description: str = "Default training configuration"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'preset': self.preset, 'data_file': self.data_file,
            'out_file': self.out_file, 'steps': self.steps,
            'tokenizer': self.tokenizer, 'vocab_size': self.vocab_size,
            'rope': self.rope, 'fresh': self.fresh,
            'block_size': self.block_size, 'n_embd': self.n_embd,
            'n_head': self.n_head, 'n_layer': self.n_layer,
            'dropout': self.dropout, 'batch_size': self.batch_size,
            'max_steps': self.max_steps, 'learning_rate': self.learning_rate,
            'eval_every': self.eval_every, 'device': self.device,
            'name': self.name, 'description': self.description,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**{k: v for k, v in config_dict.items()
                      if k in cls.__dataclass_fields__})

    def train_kwargs(self) -> Dict[str, Any]:
        """Keyword args for train.train() (None values are omitted so the
        preset's own choice wins)."""
        kw = {}
        if self.data_file:
            kw["data_file"] = self.data_file
        if self.steps:
            kw["steps"] = self.steps
        for opt in ("tokenizer", "vocab_size", "rope"):
            value = getattr(self, opt)
            if value is not None:
                kw[opt] = value
        if self.fresh:
            kw["fresh"] = True
        return kw

    def get_device(self) -> str:
        """The device train.py would use right now."""
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, 'mps') and \
                    torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device


# ============================================================================
# PRESET CONFIGURATIONS  (map onto train.py's measured ladder)
# ============================================================================

def _from_preset(preset_name: str, name: str, description: str,
                 root: str = None, **overrides) -> TrainingConfig:
    """Build a config whose architecture/training fields mirror a
    train.py preset (kept in sync for --info printing)."""
    try:
        from train import PRESETS
        cfg = PRESETS[preset_name]
    except Exception:                                   # pragma: no cover
        cfg = {}
    params = dict(
        preset=preset_name, name=name, description=description,
        block_size=cfg.get("block_size", 64),
        n_embd=cfg.get("n_embd", 96), n_head=cfg.get("n_head", 4),
        n_layer=cfg.get("n_layer", 3),
        batch_size=cfg.get("batch_size", 8),
        max_steps=cfg.get("max_steps", 500),
        learning_rate=cfg.get("lr", 3e-4),
        eval_every=cfg.get("eval_every", 100),
        tokenizer=cfg.get("tokenizer"), vocab_size=cfg.get("vocab_size"),
        rope=cfg.get("rope"),
    )
    params.update(overrides)
    return TrainingConfig(**params)


def get_coding_config() -> TrainingConfig:
    return _from_preset(
        "base", "Coding",
        "base-size model on the code corpus (override the preset's data)",
        data_file="code_corpus.txt")


def get_daily_config() -> TrainingConfig:
    return _from_preset(
        "base", "Daily Conversation",
        "base-size model on the conversation corpus")


def get_voice_config() -> TrainingConfig:
    return _from_preset(
        "nova", "Voice",
        "modern flagship (byte-level BPE 4096 + RoPE) — best quality for "
        "voice conversations; needs a GPU for long runs")


def get_balanced_config() -> TrainingConfig:
    return _from_preset(
        "beast", "Balanced",
        "10M model on the combined beast corpus (CPU-able)")


def get_tiny_config() -> TrainingConfig:
    return _from_preset("tiny", "Tiny", "0.36M starter — trains in seconds")


def get_small_config() -> TrainingConfig:
    return _from_preset("base", "Small", "2.3M — CPU fine")


def get_medium_config() -> TrainingConfig:
    return _from_preset(
        "beast", "Medium", "10M — her full-size CPU brain")


def get_large_config() -> TrainingConfig:
    return _from_preset(
        "nova", "Large", "91.3M — BPE 4096 + RoPE flagship, GPU tier")


CONFIG_REGISTRY = {
    "coding": get_coding_config,
    "daily": get_daily_config,
    "voice": get_voice_config,
    "balanced": get_balanced_config,
    "tiny": get_tiny_config,
    "small": get_small_config,
    "medium": get_medium_config,
    "large": get_large_config,
}


def get_config(config_name: str) -> TrainingConfig:
    """Get a training configuration by name.

    TrainingConfig objects can also be created from train.py preset names
    directly: get_config("xl") falls through to the ladder.
    """
    if config_name in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[config_name]()
    try:
        from train import PRESETS
        if config_name in PRESETS:
            return _from_preset(
                config_name, config_name.title(),
                f"train.py preset '{config_name}'")
    except ImportError:
        pass
    available = ", ".join(sorted(set(CONFIG_REGISTRY) |
                                 {"presets of train.py"}))
    raise ValueError(f"Unknown configuration: {config_name}. "
                     f"Available: {available}")


def list_configs() -> List[str]:
    """List all available configuration names."""
    return list(CONFIG_REGISTRY)


def print_config_info(config_name: str = None):
    """Print information about configurations."""
    if config_name:
        config = get_config(config_name)
        print(f"\n{'=' * 70}")
        print(f"  Configuration: {config.name}")
        print(f"{'=' * 70}")
        print(f"  Description: {config.description}")
        print(f"\n  Model Architecture (mirrors train.py preset "
              f"'{config.preset}'):")
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
        print(f"\n  Tokenizer: {config.tokenizer or 'char (preset default)'}"
              + (f", vocab {config.vocab_size}" if config.vocab_size else ""))
        print(f"  RoPE: {'on' if config.rope else 'off (or preset default)'}")
        print(f"  Checkpoint: {config.out_file} (ONE growing model)")
        print(f"{'=' * 70}\n")
    else:
        print(f"\n{'=' * 70}")
        print(f"  Available Training Configurations")
        print(f"{'=' * 70}")
        for name in list_configs():
            config = get_config(name)
            print(f"\n  {name}")
            print(f"    {config.description}")
            print(f"    Model: {config.n_embd} dim, {config.n_head} heads, "
                  f"{config.n_layer} layers")
            print(f"    Training: {config.batch_size} batch, "
                  f"{config.max_steps} steps")
        print(f"\n{'=' * 70}\n")


# ============================================================================
# TRAINING SCRIPT GENERATOR  (always goes through the modern trainer)
# ============================================================================

def generate_training_script(config: TrainingConfig) -> str:
    """Generate a runnable training script for the given configuration.

    The script imports the real trainer (train.train) so generated runs
    behave exactly like ``python train.py``: auto-resume one-model growth,
    CUDA/AMP/DataParallel detection, BPE + RoPE, periodic checkpoints.
    """
    kw = config.train_kwargs()
    kw_lines = ""
    for key, value in kw.items():
        kw_lines += f"    {key}={value!r},\n"
    return f'''"""Auto-generated trainer ({config.name}).

Generated by train_configs.py. Runs the REAL trainer (train.py) so the
run behaves identically to `python train.py --preset {config.preset}`:
auto-resume, periodic saves, CUDA/AMP/DataParallel, BPE + RoPE.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train

if __name__ == "__main__":
    train(
        preset="{config.preset}",
        out_path="{config.out_file}",
{kw_lines}    )
'''


def save_training_script(config: TrainingConfig, filename: str = None):
    """Save the generated training script to a file."""
    if filename is None:
        safe = config.name.lower().replace(" ", "_")
        filename = f"train_{safe}.py"
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(generate_training_script(config))
    print(f"Training script saved to {filename}")
    return filename


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Training Configuration Manager (modern-trainer layer)")
    parser.add_argument("--list", action="store_true",
                        help="List all available configurations")
    parser.add_argument("--info", type=str,
                        help="Show info about a specific configuration")
    parser.add_argument("--generate", type=str,
                        help="Generate a training script for a configuration")
    parser.add_argument("--save", type=str,
                        help="Save a training script for a configuration")

    args = parser.parse_args()

    if args.list:
        print_config_info()
    elif args.info:
        print_config_info(args.info)
    elif args.generate:
        config = get_config(args.generate)
        print(generate_training_script(config))
    elif args.save:
        config = get_config(args.save)
        save_training_script(config)
    else:
        print_config_info()

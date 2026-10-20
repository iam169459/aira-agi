"""
Causal Model - Cause and effect relationships
"""
from dataclasses import dataclass


@dataclass
class CausalRelationship:
    cause: str
    effect: str
    strength: float = 1.0


class CausalModel:
    pass

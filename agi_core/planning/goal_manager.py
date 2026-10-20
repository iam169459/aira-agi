"""
Goal Manager - Manage multiple goals and priorities
"""
from dataclasses import dataclass
from enum import Enum


class GoalState(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GoalManager:
    pass

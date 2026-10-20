"""
Meta-Learning Module
Provides self-improvement, architecture search, and curriculum learning capabilities.
"""

from .meta_learner import MetaLearner
from .self_improvement import SelfImprovementEngine
from .architecture_search import ArchitectureSearch
from .curriculum_learning import CurriculumLearner

__all__ = [
    'MetaLearner',
    'SelfImprovementEngine',
    'ArchitectureSearch',
    'CurriculumLearner',
]

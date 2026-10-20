"""
Meta-Learner - Learning to Learn

This module provides meta-learning capabilities that allow the AGI to:
- Learn from its own learning process
- Adapt its learning strategies
- Improve its knowledge acquisition efficiency
- Develop better problem-solving approaches
"""

import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
from collections import defaultdict
import hashlib


@dataclass
class LearningExperience:
    """A record of a learning experience"""
    experience_id: str
    task_description: str
    learning_method: str
    duration: float  # Seconds
    success: bool
    accuracy: float = 0.0  # 0-1
    confidence: float = 0.0  # 0-1
    lessons_learned: List[str] = field(default_factory=list)
    mistakes_made: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experience_id': self.experience_id,
            'task_description': self.task_description,
            'learning_method': self.learning_method,
            'duration': self.duration,
            'success': self.success,
            'accuracy': self.accuracy,
            'confidence': self.confidence,
            'lessons_learned': self.lessons_learned,
            'mistakes_made': self.mistakes_made,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningExperience':
        return cls(
            experience_id=data.get('experience_id', ''),
            task_description=data.get('task_description', ''),
            learning_method=data.get('learning_method', ''),
            duration=data.get('duration', 0.0),
            success=data.get('success', False),
            accuracy=data.get('accuracy', 0.0),
            confidence=data.get('confidence', 0.0),
            lessons_learned=data.get('lessons_learned', []),
            mistakes_made=data.get('mistakes_made', []),
            timestamp=data.get('timestamp', time.time())
        )


@dataclass
class LearningStrategy:
    """A learning strategy with performance metrics"""
    strategy_id: str
    name: str
    description: str
    success_rate: float = 0.0
    avg_accuracy: float = 0.0
    avg_duration: float = 0.0
    use_count: int = 0
    best_for: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def update_from_experience(self, experience: LearningExperience) -> None:
        """Update strategy metrics from a learning experience"""
        self.use_count += 1
        
        # Update success rate
        total_success = self.success_rate * (self.use_count - 1)
        if experience.success:
            total_success += 1
        self.success_rate = total_success / self.use_count
        
        # Update average accuracy
        total_accuracy = self.avg_accuracy * (self.use_count - 1)
        total_accuracy += experience.accuracy
        self.avg_accuracy = total_accuracy / self.use_count
        
        # Update average duration
        total_duration = self.avg_duration * (self.use_count - 1)
        total_duration += experience.duration
        self.avg_duration = total_duration / self.use_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'description': self.description,
            'success_rate': self.success_rate,
            'avg_accuracy': self.avg_accuracy,
            'avg_duration': self.avg_duration,
            'use_count': self.use_count,
            'best_for': self.best_for,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningStrategy':
        return cls(
            strategy_id=data.get('strategy_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            success_rate=data.get('success_rate', 0.0),
            avg_accuracy=data.get('avg_accuracy', 0.0),
            avg_duration=data.get('avg_duration', 0.0),
            use_count=data.get('use_count', 0),
            best_for=data.get('best_for', []),
            parameters=data.get('parameters', {})
        )


@dataclass
class MetaKnowledge:
    """Knowledge about knowledge and learning"""
    known_domains: List[str] = field(default_factory=list)
    learned_concepts: List[str] = field(default_factory=list)
    effective_strategies: Dict[str, List[str]] = field(default_factory=dict)
    common_mistakes: List[str] = field(default_factory=list)
    learning_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'known_domains': self.known_domains,
            'learned_concepts': self.learned_concepts,
            'effective_strategies': self.effective_strategies,
            'common_mistakes': self.common_mistakes,
            'learning_preferences': self.learning_preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaKnowledge':
        return cls(
            known_domains=data.get('known_domains', []),
            learned_concepts=data.get('learned_concepts', []),
            effective_strategies=data.get('effective_strategies', {}),
            common_mistakes=data.get('common_mistakes', []),
            learning_preferences=data.get('learning_preferences', {})
        )


class MetaLearner:
    """
    Meta-learner that learns from the AGI's learning experiences.
    
    This module enables the AGI to:
    - Track learning experiences
    - Evaluate learning strategies
    - Adapt learning approaches based on past performance
    - Identify effective strategies for different task types
    - Avoid repeating mistakes
    """
    
    def __init__(self, storage_path: str = "meta_learner.json"):
        self.storage_path = storage_path
        self.learning_experiences: Dict[str, LearningExperience] = {}
        self.learning_strategies: Dict[str, LearningStrategy] = {}
        self.meta_knowledge = MetaKnowledge()
        self.experience_history: List[str] = []
        
        # Initialize with common learning strategies
        self._initialize_strategies()
        self.load()
        
    def _initialize_strategies(self) -> None:
        """Initialize common learning strategies"""
        strategies = [
            {
                'name': 'memorization',
                'description': 'Memorize facts and information directly',
                'best_for': ['facts', 'definitions', 'simple information']
            },
            {
                'name': 'pattern_recognition',
                'description': 'Identify and learn from patterns in data',
                'best_for': ['sequences', 'repetitive tasks', 'data analysis']
            },
            {
                'name': 'analogical_reasoning',
                'description': 'Learn by drawing analogies to known concepts',
                'best_for': ['new concepts', 'problem solving', 'creative tasks']
            },
            {
                'name': 'deductive_reasoning',
                'description': 'Learn through logical deduction from first principles',
                'best_for': ['mathematics', 'logic', 'structured problems']
            },
            {
                'name': 'inductive_reasoning',
                'description': 'Learn by generalizing from specific examples',
                'best_for': ['hypothesis formation', 'prediction', 'generalization']
            },
            {
                'name': 'experimentation',
                'description': 'Learn through trial and error and experimentation',
                'best_for': ['exploration', 'discovery', 'optimization']
            },
            {
                'name': 'instruction_following',
                'description': 'Learn by following step-by-step instructions',
                'best_for': ['procedures', 'skills', 'step-by-step tasks']
            },
            {
                'name': 'collaborative_learning',
                'description': 'Learn through interaction and collaboration',
                'best_for': ['social learning', 'discussion', 'teamwork']
            }
        ]
        
        for i, strategy_data in enumerate(strategies):
            strategy = LearningStrategy(
                strategy_id=f"strategy_{i}",
                name=strategy_data['name'],
                description=strategy_data['description'],
                best_for=strategy_data['best_for']
            )
            self.learning_strategies[strategy.strategy_id] = strategy
        
    def record_experience(self, task_description: str, learning_method: str,
                          duration: float, success: bool, accuracy: float = 1.0,
                          confidence: float = 1.0, lessons: List[str] = None,
                          mistakes: List[str] = None) -> LearningExperience:
        """Record a learning experience"""
        experience_id = self._generate_id(f"exp_{learning_method}")
        
        experience = LearningExperience(
            experience_id=experience_id,
            task_description=task_description,
            learning_method=learning_method,
            duration=duration,
            success=success,
            accuracy=accuracy,
            confidence=confidence,
            lessons_learned=lessons or [],
            mistakes_made=mistakes or []
        )
        
        self.learning_experiences[experience_id] = experience
        self.experience_history.append(experience_id)
        
        # Update the corresponding strategy
        if learning_method in self.learning_strategies:
            self.learning_strategies[learning_method].update_from_experience(experience)
        
        # Update meta-knowledge
        self._update_meta_knowledge(experience)
        
        return experience
    
    def _update_meta_knowledge(self, experience: LearningExperience) -> None:
        """Update meta-knowledge based on a learning experience"""
        # Track learned concepts from task descriptions
        words = experience.task_description.lower().split()
        for word in words:
            if len(word) > 4 and word not in self.meta_knowledge.learned_concepts:
                self.meta_knowledge.learned_concepts.append(word)
        
        # Track mistakes
        for mistake in experience.mistakes_made:
            if mistake not in self.meta_knowledge.common_mistakes:
                self.meta_knowledge.common_mistakes.append(mistake)
        
        # Update effective strategies
        if experience.success:
            task_type = self._infer_task_type(experience.task_description)
            if task_type not in self.meta_knowledge.effective_strategies:
                self.meta_knowledge.effective_strategies[task_type] = []
            if experience.learning_method not in self.meta_knowledge.effective_strategies[task_type]:
                self.meta_knowledge.effective_strategies[task_type].append(experience.learning_method)
    
    def _infer_task_type(self, task_description: str) -> str:
        """Infer the type of task from its description"""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['fact', 'definition', 'remember', 'memorize']):
            return 'memorization'
        elif any(word in task_lower for word in ['pattern', 'sequence', 'repeat']):
            return 'pattern_recognition'
        elif any(word in task_lower for word in ['analogy', 'similar', 'like']):
            return 'analogical_reasoning'
        elif any(word in task_lower for word in ['logic', 'deduce', 'prove', 'math']):
            return 'deductive_reasoning'
        elif any(word in task_lower for word in ['generalize', 'predict', 'hypothesis']):
            return 'inductive_reasoning'
        elif any(word in task_lower for word in ['try', 'experiment', 'test']):
            return 'experimentation'
        elif any(word in task_lower for word in ['instruct', 'follow', 'step']):
            return 'instruction_following'
        else:
            return 'general'
    
    def get_best_strategy(self, task_description: str = None) -> Optional[LearningStrategy]:
        """Get the best learning strategy for a given task"""
        if task_description:
            task_type = self._infer_task_type(task_description)
            
            # Check if we have effective strategies for this task type
            if task_type in self.meta_knowledge.effective_strategies:
                strategy_names = self.meta_knowledge.effective_strategies[task_type]
                best_strategy = None
                best_score = -1.0
                
                for strategy_name in strategy_names:
                    for strategy in self.learning_strategies.values():
                        if strategy.name == strategy_name:
                            # Score based on success rate and relevance
                            score = strategy.success_rate * 0.7 + strategy.avg_accuracy * 0.3
                            if score > best_score:
                                best_score = score
                                best_strategy = strategy
                
                if best_strategy:
                    return best_strategy
        
        # Fall back to the overall best strategy
        return self.get_overall_best_strategy()
    
    def get_overall_best_strategy(self) -> Optional[LearningStrategy]:
        """Get the overall best learning strategy"""
        best_strategy = None
        best_score = -1.0
        
        for strategy in self.learning_strategies.values():
            if strategy.use_count == 0:
                continue
            score = strategy.success_rate * 0.7 + strategy.avg_accuracy * 0.3
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy
    
    def get_strategy_recommendations(self, task_description: str) -> List[Dict[str, Any]]:
        """Get recommendations for learning strategies for a task"""
        recommendations = []
        
        # Get the best strategy
        best = self.get_best_strategy(task_description)
        if best:
            recommendations.append({
                'strategy': best.name,
                'description': best.description,
                'success_rate': best.success_rate,
                'avg_accuracy': best.avg_accuracy,
                'reason': 'Best match for task type'
            })
        
        # Add alternative strategies
        task_type = self._infer_task_type(task_description) if task_description else 'general'
        for strategy in self.learning_strategies.values():
            if strategy.name != best.name and strategy.use_count > 0:
                if task_type in strategy.best_for or 'general' in strategy.best_for:
                    recommendations.append({
                        'strategy': strategy.name,
                        'description': strategy.description,
                        'success_rate': strategy.success_rate,
                        'avg_accuracy': strategy.avg_accuracy,
                        'reason': 'Alternative approach'
                    })
        
        # Sort by success rate
        recommendations.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return recommendations[:3]  # Return top 3
    
    def learn_from_mistakes(self) -> List[str]:
        """Analyze common mistakes and suggest improvements"""
        suggestions = []
        
        # Count mistake frequency
        mistake_counts = defaultdict(int)
        for experience in self.learning_experiences.values():
            for mistake in experience.mistakes_made:
                mistake_counts[mistake] += 1
        
        # Generate suggestions for frequent mistakes
        for mistake, count in mistake_counts.items():
            if count > 1:  # Only suggest for repeated mistakes
                suggestions.append(f"Avoid: {mistake} (occurred {count} times)")
        
        return suggestions
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning experiences"""
        total_experiences = len(self.learning_experiences)
        successful = sum(1 for e in self.learning_experiences.values() if e.success)
        
        return {
            'total_experiences': total_experiences,
            'success_rate': successful / total_experiences if total_experiences > 0 else 0.0,
            'total_strategies': len(self.learning_strategies),
            'learned_concepts': len(self.meta_knowledge.learned_concepts),
            'common_mistakes': len(self.meta_knowledge.common_mistakes),
            'best_strategy': self.get_overall_best_strategy().name if self.get_overall_best_strategy() else None
        }
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(str(timestamp).encode()).hexdigest()[:4]
        return f"{prefix}_{timestamp}_{random_part}"
    
    def save(self) -> None:
        """Save meta-learner state to file"""
        state = {
            'learning_experiences': {k: v.to_dict() for k, v in self.learning_experiences.items()},
            'learning_strategies': {k: v.to_dict() for k, v in self.learning_strategies.items()},
            'meta_knowledge': self.meta_knowledge.to_dict(),
            'experience_history': self.experience_history
        }
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self) -> bool:
        """Load meta-learner state from file"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    state = json.load(f)
                
                self.learning_experiences = {
                    k: LearningExperience.from_dict(v) for k, v in state.get('learning_experiences', {}).items()
                }
                self.learning_strategies = {
                    k: LearningStrategy.from_dict(v) for k, v in state.get('learning_strategies', {}).items()
                }
                self.meta_knowledge = MetaKnowledge.from_dict(state.get('meta_knowledge', {}))
                self.experience_history = state.get('experience_history', [])
                
                return True
        except FileNotFoundError:
            pass
        return False
    
    def clear(self) -> None:
        """Clear all learning data"""
        self.learning_experiences.clear()
        self.experience_history.clear()
        self.meta_knowledge = MetaKnowledge()

"""
Reflection Engine - Meta-Cognition and Self-Improvement

This module provides capabilities for the AGI to reflect on its own
thinking processes, identify errors, and improve future reasoning.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import json


class ReflectionType(Enum):
    """Types of reflection the system can perform"""
    SELF_CRITIQUE = "self_critique"
    META_COGNITION = "meta_cognition"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    ERROR_ANALYSIS = "error_analysis"
    LEARNING_REFLECTION = "learning_reflection"
    STRATEGY_EVALUATION = "strategy_evaluation"


class ReflectionPhase(Enum):
    """Phases of the reflection process"""
    IDENTIFICATION = "identification"
    ANALYSIS = "analysis"
    EVALUATION = "evaluation"
    IMPROVEMENT = "improvement"
    IMPLEMENTATION = "implementation"


@dataclass
class ReflectionPoint:
    """A specific point of reflection on past reasoning"""
    reflection_id: str
    reflection_type: ReflectionType
    target: str  # What we're reflecting on (chain_id, decision_id, etc.)
    observation: str
    analysis: str
    evaluation: str
    improvement: str
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    related_reflections: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'reflection_id': self.reflection_id,
            'reflection_type': self.reflection_type.value,
            'target': self.target,
            'observation': self.observation,
            'analysis': self.analysis,
            'evaluation': self.evaluation,
            'improvement': self.improvement,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'related_reflections': self.related_reflections
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionPoint':
        return cls(
            reflection_id=data.get('reflection_id', ''),
            reflection_type=ReflectionType(data.get('reflection_type', 'self_critique')),
            target=data.get('target', ''),
            observation=data.get('observation', ''),
            analysis=data.get('analysis', ''),
            evaluation=data.get('evaluation', ''),
            improvement=data.get('improvement', ''),
            confidence=data.get('confidence', 0.0),
            timestamp=data.get('timestamp', time.time()),
            related_reflections=data.get('related_reflections', [])
        )


@dataclass
class SelfCritique:
    """Structured self-critique of reasoning or action"""
    critique_id: str
    subject: str  # What is being critiqued
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    overall_score: float = 0.0  # 0-100
    timestamp: float = field(default_factory=time.time)
    
    def add_strength(self, strength: str) -> None:
        self.strengths.append(strength)
    
    def add_weakness(self, weakness: str) -> None:
        self.weaknesses.append(weakness)
    
    def add_error(self, error: str) -> None:
        self.errors.append(error)
    
    def add_suggestion(self, suggestion: str) -> None:
        self.suggestions.append(suggestion)
    
    def calculate_score(self) -> float:
        """Calculate overall score based on strengths, weaknesses, and errors"""
        score = 50.0  # Start at middle
        
        # Strengths add to score
        score += len(self.strengths) * 5.0
        
        # Weaknesses subtract from score
        score -= len(self.weaknesses) * 3.0
        
        # Errors significantly subtract from score
        score -= len(self.errors) * 10.0
        
        # Clamp to 0-100
        self.overall_score = max(0.0, min(100.0, score))
        return self.overall_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'critique_id': self.critique_id,
            'subject': self.subject,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'errors': self.errors,
            'suggestions': self.suggestions,
            'overall_score': self.overall_score,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfCritique':
        critique = cls(
            critique_id=data.get('critique_id', ''),
            subject=data.get('subject', ''),
            strengths=data.get('strengths', []),
            weaknesses=data.get('weaknesses', []),
            errors=data.get('errors', []),
            suggestions=data.get('suggestions', []),
            overall_score=data.get('overall_score', 0.0),
            timestamp=data.get('timestamp', time.time())
        )
        return critique


@dataclass
class MetaCognition:
    """Meta-cognitive state and capabilities"""
    thinking_about_thinking: bool = False
    current_focus: str = ""
    cognitive_load: float = 0.0  # 0-1
    uncertainty: float = 0.0  # 0-1
    confidence: float = 1.0  # 0-1
    
    def update_state(self, focus: str, load: float, uncertainty: float) -> None:
        """Update the meta-cognitive state"""
        self.current_focus = focus
        self.cognitive_load = max(0.0, min(1.0, load))
        self.uncertainty = max(0.0, min(1.0, uncertainty))
        self.confidence = max(0.0, min(1.0, 1.0 - uncertainty * 0.5))
    
    def should_reflect(self, threshold: float = 0.7) -> bool:
        """Determine if reflection is needed based on uncertainty"""
        return self.uncertainty > threshold or self.cognitive_load > threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'thinking_about_thinking': self.thinking_about_thinking,
            'current_focus': self.current_focus,
            'cognitive_load': self.cognitive_load,
            'uncertainty': self.uncertainty,
            'confidence': self.confidence
        }


class ReflectionEngine:
    """
    Engine for performing meta-cognitive reflection on reasoning processes.
    
    Capabilities:
    - Self-critique of reasoning chains
    - Meta-cognitive monitoring
    - Performance analysis
    - Error pattern detection
    - Learning from mistakes
    """
    
    def __init__(self):
        self.reflections: Dict[str, ReflectionPoint] = {}
        self.critiques: Dict[str, SelfCritique] = {}
        self.meta_cognition = MetaCognition()
        self.reflection_history: List[str] = []
        
    def create_reflection(self, reflection_type: ReflectionType,
                          target: str, observation: str) -> ReflectionPoint:
        """Create a new reflection point"""
        reflection_id = f"reflect_{int(time.time())}_{len(self.reflections)}"
        reflection = ReflectionPoint(
            reflection_id=reflection_id,
            reflection_type=reflection_type,
            target=target,
            observation=observation
        )
        self.reflections[reflection_id] = reflection
        self.reflection_history.append(reflection_id)
        return reflection
    
    def perform_self_critique(self, subject: str, 
                             strengths: List[str] = None,
                             weaknesses: List[str] = None,
                             errors: List[str] = None) -> SelfCritique:
        """Perform a structured self-critique"""
        critique_id = f"critique_{int(time.time())}_{len(self.critiques)}"
        critique = SelfCritique(
            critique_id=critique_id,
            subject=subject,
            strengths=strengths or [],
            weaknesses=weaknesses or [],
            errors=errors or []
        )
        critique.calculate_score()
        self.critiques[critique_id] = critique
        return critique
    
    def critique_reasoning_chain(self, chain, 
                                  expected_answer: Optional[str] = None) -> SelfCritique:
        """Critique a reasoning chain"""
        subject = f"Chain: {chain.goal}"
        critique = self.perform_self_critique(subject)
        
        # Analyze the chain
        if len(chain.steps) < 3:
            critique.add_weakness("Chain is too short - may lack depth")
        else:
            critique.add_strength(f"Chain has {len(chain.steps)} steps - good depth")
        
        if chain.confidence < 0.7:
            critique.add_weakness(f"Low overall confidence: {chain.confidence:.2f}")
        else:
            critique.add_strength(f"High confidence: {chain.confidence:.2f}")
        
        if chain.contradictions:
            for contradiction in chain.contradictions:
                critique.add_error(f"Contradiction detected: {contradiction}")
        
        # Check if conclusion matches expected
        if expected_answer and chain.get_conclusion():
            if expected_answer.lower() in chain.get_conclusion().lower():
                critique.add_strength("Conclusion matches expected answer")
            else:
                critique.add_weakness("Conclusion doesn't match expected answer")
        
        # Add suggestions
        if chain.confidence < 0.8:
            critique.add_suggestion("Consider gathering more evidence or information")
        
        if len(chain.steps) > 10:
            critique.add_suggestion("Consider breaking into sub-chains for clarity")
        
        critique.calculate_score()
        return critique
    
    def reflect_on_decision(self, decision, outcome: str, 
                           expected_outcome: Optional[str] = None) -> ReflectionPoint:
        """Reflect on a decision and its outcome"""
        reflection = self.create_reflection(
            reflection_type=ReflectionType.PERFORMANCE_ANALYSIS,
            target=decision,
            observation=f"Decision: {decision}, Outcome: {outcome}"
        )
        
        # Analyze the decision
        if expected_outcome:
            if outcome == expected_outcome:
                reflection.analysis = "Decision achieved expected outcome"
                reflection.evaluation = "Good decision"
            else:
                reflection.analysis = f"Decision did not achieve expected outcome. Expected: {expected_outcome}, Got: {outcome}"
                reflection.evaluation = "Needs improvement"
                reflection.improvement = "Consider alternative approaches or gather more information before deciding"
        else:
            reflection.analysis = f"Decision made without clear expectations. Outcome: {outcome}"
            reflection.evaluation = "Neutral - no baseline for comparison"
            reflection.improvement = "Establish expected outcomes before making decisions"
        
        reflection.confidence = 0.8
        return reflection
    
    def detect_error_patterns(self, recent_errors: List[str]) -> Dict[str, Any]:
        """Detect patterns in recent errors"""
        error_counts = defaultdict(int)
        for error in recent_errors:
            # Simple pattern detection
            error_lower = error.lower()
            if "typeerror" in error_lower:
                error_counts["type_errors"] += 1
            elif "valueerror" in error_lower:
                error_counts["value_errors"] += 1
            elif "index" in error_lower:
                error_counts["index_errors"] += 1
            elif "keyerror" in error_lower:
                error_counts["key_errors"] += 1
            elif "import" in error_lower:
                error_counts["import_errors"] += 1
            else:
                error_counts["other_errors"] += 1
        
        return {
            'patterns': dict(error_counts),
            'total_errors': len(recent_errors),
            'most_common': max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        }
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """Get statistics about reflections performed"""
        return {
            'total_reflections': len(self.reflections),
            'total_critiques': len(self.critiques),
            'recent_reflections': self.reflection_history[-10:] if self.reflection_history else [],
            'meta_cognition_state': self.meta_cognition.to_dict()
        }
    
    def update_meta_cognition(self, focus: str, load: float, uncertainty: float) -> None:
        """Update the meta-cognitive state"""
        self.meta_cognition.update_state(focus, load, uncertainty)
    
    def should_trigger_reflection(self, uncertainty: float = 0.0, 
                                  cognitive_load: float = 0.0) -> bool:
        """Determine if reflection should be triggered"""
        self.update_meta_cognition("", cognitive_load, uncertainty)
        return self.meta_cognition.should_reflect()
    
    def save_state(self, path: str = "agi_reflections.json") -> None:
        """Save reflection state to file"""
        state = {
            'reflections': {k: v.to_dict() for k, v in self.reflections.items()},
            'critiques': {k: v.to_dict() for k, v in self.critiques.items()},
            'meta_cognition': self.meta_cognition.to_dict(),
            'reflection_history': self.reflection_history
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str = "agi_reflections.json") -> bool:
        """Load reflection state from file"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.reflections = {k: ReflectionPoint.from_dict(v) for k, v in state.get('reflections', {}).items()}
            self.critiques = {k: SelfCritique.from_dict(v) for k, v in state.get('critiques', {}).items()}
            self.reflection_history = state.get('reflection_history', [])
            
            meta_data = state.get('meta_cognition', {})
            self.meta_cognition = MetaCognition(
                thinking_about_thinking=meta_data.get('thinking_about_thinking', False),
                current_focus=meta_data.get('current_focus', ""),
                cognitive_load=meta_data.get('cognitive_load', 0.0),
                uncertainty=meta_data.get('uncertainty', 0.0),
                confidence=meta_data.get('confidence', 1.0)
            )
            return True
        except FileNotFoundError:
            return False

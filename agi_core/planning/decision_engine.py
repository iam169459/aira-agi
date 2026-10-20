"""
Decision Engine - Multi-Criteria Decision Making

This module provides the AGI's decision-making capabilities, including:
- Option evaluation
- Multi-criteria decision analysis
- Cost-benefit analysis
- Risk assessment
- Decision validation
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
from enum import Enum
import math


class DecisionStrategy(Enum):
    """Different decision-making strategies"""
    MAXIMIZE_UTILITY = "maximize_utility"
    MINIMIZE_REGRET = "minimize_regret"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_CONFIDENCE = "maximize_confidence"
    BALANCED = "balanced"
    CONSENSUS = "consensus"


class DecisionStatus(Enum):
    """Status of a decision"""
    PENDING = "pending"
    EVALUATING = "evaluating"
    DECIDED = "decided"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class Criterion:
    """A criterion for evaluating options"""
    name: str
    description: str
    weight: float = 1.0  # 0-1, importance of this criterion
    is_maximizing: bool = True  # True if higher values are better
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'weight': self.weight,
            'is_maximizing': self.is_maximizing
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Criterion':
        return cls(
            name=data.get('name', ''),
            description=data.get('description', ''),
            weight=data.get('weight', 1.0),
            is_maximizing=data.get('is_maximizing', True)
        )


@dataclass
class Option:
    """An option to be evaluated in a decision"""
    option_id: str
    description: str
    scores: Dict[str, float] = field(default_factory=dict)  # criterion_name -> score
    confidence: float = 1.0
    probability: float = 1.0  # Probability of success
    cost: float = 0.0
    benefits: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def get_score(self, criterion: Criterion) -> float:
        """Get the score for a specific criterion"""
        return self.scores.get(criterion.name, 0.0)
    
    def set_score(self, criterion_name: str, score: float) -> None:
        """Set the score for a criterion"""
        self.scores[criterion_name] = score
    
    def get_weighted_score(self, criteria: List[Criterion]) -> float:
        """Calculate weighted score across all criteria"""
        total_score = 0.0
        total_weight = 0.0
        
        for criterion in criteria:
            score = self.get_score(criterion)
            if not criterion.is_maximizing:
                # For minimizing criteria, invert the score
                score = 1.0 - score
            
            total_score += score * criterion.weight
            total_weight += criterion.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_utility(self, criteria: List[Criterion], 
                    risk_weight: float = 0.3) -> float:
        """Calculate overall utility considering benefits, costs, and risks"""
        # Weighted score from criteria
        weighted_score = self.get_weighted_score(criteria)
        
        # Adjust for probability of success
        adjusted_score = weighted_score * self.probability
        
        # Adjust for cost (higher cost reduces utility)
        cost_factor = 1.0 / (1.0 + self.cost * 0.1)  # Normalize cost impact
        
        # Adjust for risks
        risk_count = len(self.risks)
        risk_factor = 1.0 - (risk_count * 0.1 * risk_weight)
        
        # Combine factors
        utility = adjusted_score * cost_factor * risk_factor
        
        return max(0.0, min(1.0, utility))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'option_id': self.option_id,
            'description': self.description,
            'scores': self.scores,
            'confidence': self.confidence,
            'probability': self.probability,
            'cost': self.cost,
            'benefits': self.benefits,
            'risks': self.risks,
            'dependencies': self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Option':
        return cls(
            option_id=data.get('option_id', ''),
            description=data.get('description', ''),
            scores=data.get('scores', {}),
            confidence=data.get('confidence', 1.0),
            probability=data.get('probability', 1.0),
            cost=data.get('cost', 0.0),
            benefits=data.get('benefits', []),
            risks=data.get('risks', []),
            dependencies=data.get('dependencies', [])
        )


@dataclass
class Decision:
    """A decision to be made"""
    decision_id: str
    description: str
    context: str = ""
    options: List[Option] = field(default_factory=list)
    criteria: List[Criterion] = field(default_factory=list)
    strategy: DecisionStrategy = DecisionStrategy.BALANCED
    status: DecisionStatus = DecisionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None
    executed_at: Optional[float] = None
    chosen_option: Optional[Option] = None
    reasoning: str = ""
    
    def add_option(self, option: Option) -> None:
        """Add an option to the decision"""
        self.options.append(option)
    
    def add_criterion(self, criterion: Criterion) -> None:
        """Add a criterion for evaluation"""
        self.criteria.append(criterion)
    
    def evaluate_options(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all options against all criteria"""
        results = {}
        
        for option in self.options:
            scores = {}
            for criterion in self.criteria:
                score = option.get_score(criterion)
                if not criterion.is_maximizing:
                    score = 1.0 - score
                scores[criterion.name] = score * criterion.weight
            results[option.option_id] = scores
        
        return results
    
    def make_decision(self) -> Optional[Option]:
        """Make the optimal decision based on the current strategy"""
        if not self.options or not self.criteria:
            return None
        
        self.status = DecisionStatus.EVALUATING
        
        # Evaluate all options
        evaluations = self.evaluate_options()
        
        # Apply the decision strategy
        if self.strategy == DecisionStrategy.MAXIMIZE_UTILITY:
            best_option = self._maximize_utility()
        elif self.strategy == DecisionStrategy.MINIMIZE_REGRET:
            best_option = self._minimize_regret()
        elif self.strategy == DecisionStrategy.MINIMIZE_RISK:
            best_option = self._minimize_risk()
        elif self.strategy == DecisionStrategy.MAXIMIZE_CONFIDENCE:
            best_option = self._maximize_confidence()
        elif self.strategy == DecisionStrategy.CONSENSUS:
            best_option = self._consensus()
        else:  # BALANCED
            best_option = self._balanced_decision()
        
        if best_option:
            self.chosen_option = best_option
            self.status = DecisionStatus.DECIDED
            self.decided_at = time.time()
            self.reasoning = f"Chose {best_option.description} using {self.strategy.value} strategy"
        
        return best_option
    
    def _maximize_utility(self) -> Optional[Option]:
        """Choose the option with maximum utility"""
        best_option = None
        best_utility = -1.0
        
        for option in self.options:
            utility = option.get_utility(self.criteria)
            if utility > best_utility:
                best_utility = utility
                best_option = option
        
        return best_option
    
    def _minimize_regret(self) -> Optional[Option]:
        """Choose the option that minimizes potential regret"""
        # Calculate regret as the difference between the best possible outcome
        # and the chosen option's outcome
        best_option = None
        min_regret = float('inf')
        
        # Find the best possible score for each criterion
        best_scores = {}
        for criterion in self.criteria:
            best_score = max([o.get_score(criterion) for o in self.options])
            best_scores[criterion.name] = best_score
        
        for option in self.options:
            # Calculate regret for this option
            regret = 0.0
            for criterion in self.criteria:
                option_score = option.get_score(criterion)
                best_score = best_scores[criterion.name]
                
                if criterion.is_maximizing:
                    regret += (best_score - option_score) * criterion.weight
                else:
                    # For minimizing criteria, regret is higher when score is lower
                    regret += (option_score - best_score) * criterion.weight
            
            if regret < min_regret:
                min_regret = regret
                best_option = option
        
        return best_option
    
    def _minimize_risk(self) -> Optional[Option]:
        """Choose the option with the least risk"""
        best_option = None
        min_risk = float('inf')
        
        for option in self.options:
            # Calculate risk score (lower is better)
            risk_score = len(option.risks) * (1.0 - option.probability)
            
            if risk_score < min_risk:
                min_risk = risk_score
                best_option = option
        
        return best_option
    
    def _maximize_confidence(self) -> Optional[Option]:
        """Choose the option with the highest confidence"""
        best_option = None
        best_confidence = -1.0
        
        for option in self.options:
            if option.confidence > best_confidence:
                best_confidence = option.confidence
                best_option = option
        
        return best_option
    
    def _consensus(self) -> Optional[Option]:
        """Choose the option that performs well across all criteria"""
        # Calculate consensus score based on variance across criteria
        best_option = None
        best_consensus = -1.0
        
        for option in self.options:
            # Calculate scores for all criteria
            scores = [option.get_score(c) for c in self.criteria]
            
            if not scores:
                continue
            
            # Calculate mean and variance
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            
            # Consensus score: high mean, low variance
            consensus_score = mean_score - variance * 0.5
            
            if consensus_score > best_consensus:
                best_consensus = consensus_score
                best_option = option
        
        return best_option
    
    def _balanced_decision(self) -> Optional[Option]:
        """Make a balanced decision considering multiple factors"""
        best_option = None
        best_score = -1.0
        
        for option in self.options:
            # Calculate utility
            utility = option.get_utility(self.criteria)
            
            # Adjust for confidence
            confidence_factor = option.confidence
            
            # Adjust for cost
            cost_factor = 1.0 / (1.0 + option.cost * 0.05)
            
            # Adjust for probability
            probability_factor = option.probability
            
            # Combined score
            combined_score = utility * confidence_factor * cost_factor * probability_factor
            
            if combined_score > best_score:
                best_score = combined_score
                best_option = option
        
        return best_option
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'description': self.description,
            'context': self.context,
            'options': [o.to_dict() for o in self.options],
            'criteria': [c.to_dict() for c in self.criteria],
            'strategy': self.strategy.value,
            'status': self.status.value,
            'created_at': self.created_at,
            'decided_at': self.decided_at,
            'executed_at': self.executed_at,
            'chosen_option_id': self.chosen_option.option_id if self.chosen_option else None,
            'reasoning': self.reasoning
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Decision':
        decision = cls(
            decision_id=data.get('decision_id', ''),
            description=data.get('description', ''),
            context=data.get('context', ''),
            options=[Option.from_dict(o) for o in data.get('options', [])],
            criteria=[Criterion.from_dict(c) for c in data.get('criteria', [])],
            strategy=DecisionStrategy(data.get('strategy', 'balanced')),
            status=DecisionStatus(data.get('status', 'pending')),
            created_at=data.get('created_at', time.time()),
            decided_at=data.get('decided_at'),
            executed_at=data.get('executed_at'),
            reasoning=data.get('reasoning', '')
        )
        
        # Set chosen option
        chosen_option_id = data.get('chosen_option_id')
        if chosen_option_id:
            for option in decision.options:
                if option.option_id == chosen_option_id:
                    decision.chosen_option = option
                    break
        
        return decision


class DecisionEngine:
    """
    Engine for making decisions using multiple strategies and criteria.
    
    This engine can:
    - Define and evaluate decision criteria
    - Generate and evaluate options
    - Make optimal decisions based on selected strategies
    - Track decision outcomes and learn from them
    """
    
    def __init__(self):
        self.decisions: Dict[str, Decision] = {}
        self.criteria_library: Dict[str, Criterion] = {}
        self.decision_history: List[str] = []
        self.outcome_history: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with common criteria
        self._initialize_criteria_library()
        
    def _initialize_criteria_library(self) -> None:
        """Initialize the criteria library with common criteria"""
        # Effectiveness criteria
        self.add_criterion(Criterion(
            name="effectiveness",
            description="How well the option achieves the desired outcome",
            weight=1.0,
            is_maximizing=True
        ))
        
        # Efficiency criteria
        self.add_criterion(Criterion(
            name="efficiency",
            description="How efficiently the option achieves the outcome",
            weight=0.8,
            is_maximizing=True
        ))
        
        # Cost criteria (minimizing)
        self.add_criterion(Criterion(
            name="cost",
            description="The cost of executing the option",
            weight=0.7,
            is_maximizing=False
        ))
        
        # Risk criteria (minimizing)
        self.add_criterion(Criterion(
            name="risk",
            description="The risk associated with the option",
            weight=0.8,
            is_maximizing=False
        ))
        
        # Confidence criteria
        self.add_criterion(Criterion(
            name="confidence",
            description="Confidence in the success of the option",
            weight=0.6,
            is_maximizing=True
        ))
        
        # Time criteria (minimizing)
        self.add_criterion(Criterion(
            name="time",
            description="Time required to execute the option",
            weight=0.5,
            is_maximizing=False
        ))
    
    def add_criterion(self, criterion: Criterion) -> None:
        """Add a criterion to the library"""
        self.criteria_library[criterion.name] = criterion
    
    def get_criterion(self, name: str) -> Optional[Criterion]:
        """Get a criterion by name"""
        return self.criteria_library.get(name)
    
    def create_decision(self, description: str, context: str = "",
                        decision_id: Optional[str] = None) -> Decision:
        """Create a new decision"""
        decision_id = decision_id or f"dec_{int(time.time() * 1000)}"
        
        decision = Decision(
            decision_id=decision_id,
            description=description,
            context=context
        )
        
        self.decisions[decision_id] = decision
        return decision
    
    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Get a decision by ID"""
        return self.decisions.get(decision_id)
    
    def make_decision(self, description: str, options: List[str],
                      context: str = "", strategy: DecisionStrategy = None,
                      criteria_names: List[str] = None) -> Decision:
        """
        Make a decision given a description, options, and context.
        
        This is a high-level interface for decision making.
        """
        # Create decision
        decision = self.create_decision(description, context)
        
        # Add criteria
        if criteria_names:
            for name in criteria_names:
                criterion = self.get_criterion(name)
                if criterion:
                    decision.add_criterion(criterion)
        else:
            # Use default criteria
            for criterion in self.criteria_library.values():
                decision.add_criterion(criterion)
        
        # Add options
        for i, option_desc in enumerate(options):
            option = Option(
                option_id=f"opt_{decision.decision_id}_{i}",
                description=option_desc
            )
            decision.add_option(option)
        
        # Set strategy
        if strategy:
            decision.strategy = strategy
        
        # Make the decision
        decision.make_decision()
        
        return decision
    
    def record_outcome(self, decision_id: str, outcome: str,
                       success: bool = True, metrics: Dict[str, float] = None) -> None:
        """Record the outcome of a decision"""
        if decision_id in self.outcome_history:
            # Update existing
            self.outcome_history[decision_id].update({
                'outcome': outcome,
                'success': success,
                'metrics': metrics or {},
                'timestamp': time.time()
            })
        else:
            # Create new
            self.outcome_history[decision_id] = {
                'outcome': outcome,
                'success': success,
                'metrics': metrics or {},
                'timestamp': time.time()
            }
        
        self.decision_history.append(decision_id)
    
    def learn_from_outcomes(self) -> Dict[str, Any]:
        """
        Learn from recorded decision outcomes to improve future decisions.
        
        This would adjust criteria weights and decision strategies based on
        which decisions led to successful outcomes.
        """
        # Count successful vs failed decisions by strategy
        strategy_success = defaultdict(lambda: {'success': 0, 'total': 0})
        
        for decision_id, outcome in self.outcome_history.items():
            if decision_id in self.decisions:
                decision = self.decisions[decision_id]
                strategy_success[decision.strategy.value]['total'] += 1
                if outcome.get('success', False):
                    strategy_success[decision.strategy.value]['success'] += 1
        
        # Calculate success rates
        strategy_stats = {}
        for strategy, counts in strategy_success.items():
            success_rate = counts['success'] / counts['total'] if counts['total'] > 0 else 0.0
            strategy_stats[strategy] = {
                'success_rate': success_rate,
                'total': counts['total']
            }
        
        return {
            'strategy_stats': strategy_stats,
            'total_decisions': len(self.decision_history),
            'successful_decisions': sum(1 for o in self.outcome_history.values() if o.get('success', False))
        }
    
    def get_best_strategy(self) -> DecisionStrategy:
        """Get the strategy with the highest success rate"""
        stats = self.learn_from_outcomes()
        best_strategy = None
        best_rate = -1.0
        
        for strategy, data in stats['strategy_stats'].items():
            if data['success_rate'] > best_rate:
                best_rate = data['success_rate']
                best_strategy = DecisionStrategy(strategy)
        
        return best_strategy or DecisionStrategy.BALANCED
    
    def clear(self) -> None:
        """Clear all decisions and history"""
        self.decisions.clear()
        self.decision_history.clear()
        self.outcome_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get decision engine statistics"""
        return {
            'total_decisions': len(self.decisions),
            'total_criteria': len(self.criteria_library),
            'decision_history_length': len(self.decision_history),
            'outcome_history_length': len(self.outcome_history),
            'best_strategy': self.get_best_strategy().value
        }

"""
Planner - Core Planning and Goal Achievement System

This module provides the AGI's planning capabilities, including:
- Goal definition and management
- Action planning and sequencing
- Plan execution and monitoring
- Plan adaptation and repair
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Set, Tuple
from enum import Enum
import hashlib


class GoalStatus(Enum):
    """Status of a goal"""
    PENDING = "pending"
    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ActionStatus(Enum):
    """Status of an action"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(Enum):
    """Status of a plan"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Goal:
    """A goal that the AGI is trying to achieve"""
    goal_id: str
    description: str
    priority: float = 0.5  # 0-1, higher is more important
    deadline: Optional[float] = None  # Unix timestamp
    status: GoalStatus = GoalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)  # Other goal IDs
    subgoals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    failure_criteria: List[str] = field(default_factory=list)
    progress: float = 0.0  # 0-1
    
    def is_achieved(self) -> bool:
        """Check if goal is achieved"""
        return self.status == GoalStatus.ACHIEVED
    
    def is_active(self) -> bool:
        """Check if goal is currently active"""
        return self.status == GoalStatus.ACTIVE
    
    def is_pending(self) -> bool:
        """Check if goal is pending"""
        return self.status == GoalStatus.PENDING
    
    def is_failed(self) -> bool:
        """Check if goal has failed"""
        return self.status == GoalStatus.FAILED
    
    def update_progress(self, new_progress: float) -> None:
        """Update the progress of this goal"""
        self.progress = max(0.0, min(1.0, new_progress))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goal_id': self.goal_id,
            'description': self.description,
            'priority': self.priority,
            'deadline': self.deadline,
            'status': self.status.value,
            'created_at': self.created_at,
            'dependencies': self.dependencies,
            'subgoals': self.subgoals,
            'constraints': self.constraints,
            'success_criteria': self.success_criteria,
            'failure_criteria': self.failure_criteria,
            'progress': self.progress
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        return cls(
            goal_id=data.get('goal_id', ''),
            description=data.get('description', ''),
            priority=data.get('priority', 0.5),
            deadline=data.get('deadline'),
            status=GoalStatus(data.get('status', 'pending')),
            created_at=data.get('created_at', time.time()),
            dependencies=data.get('dependencies', []),
            subgoals=data.get('subgoals', []),
            constraints=data.get('constraints', []),
            success_criteria=data.get('success_criteria', []),
            failure_criteria=data.get('failure_criteria', []),
            progress=data.get('progress', 0.0)
        )


@dataclass
class Action:
    """An action that can be executed to achieve a goal"""
    action_id: str
    name: str
    description: str
    goal_id: str  # The goal this action contributes to
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 0.0  # Seconds
    estimated_cost: float = 0.0
    priority: float = 0.5
    status: ActionStatus = ActionStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    
    def is_completed(self) -> bool:
        """Check if action is completed"""
        return self.status == ActionStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if action has failed"""
        return self.status == ActionStatus.FAILED
    
    def is_executing(self) -> bool:
        """Check if action is currently executing"""
        return self.status == ActionStatus.EXECUTING
    
    def execute(self, executor: Callable) -> bool:
        """Execute this action using the provided executor"""
        self.status = ActionStatus.EXECUTING
        self.start_time = time.time()
        
        try:
            result = executor(self)
            self.status = ActionStatus.COMPLETED
            self.end_time = time.time()
            self.result = str(result)
            return True
        except Exception as e:
            self.status = ActionStatus.FAILED
            self.end_time = time.time()
            self.error = str(e)
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'name': self.name,
            'description': self.description,
            'goal_id': self.goal_id,
            'preconditions': self.preconditions,
            'postconditions': self.postconditions,
            'parameters': self.parameters,
            'estimated_duration': self.estimated_duration,
            'estimated_cost': self.estimated_cost,
            'priority': self.priority,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'result': self.result,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        return cls(
            action_id=data.get('action_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            goal_id=data.get('goal_id', ''),
            preconditions=data.get('preconditions', []),
            postconditions=data.get('postconditions', []),
            parameters=data.get('parameters', {}),
            estimated_duration=data.get('estimated_duration', 0.0),
            estimated_cost=data.get('estimated_cost', 0.0),
            priority=data.get('priority', 0.5),
            status=ActionStatus(data.get('status', 'pending')),
            start_time=data.get('start_time'),
            end_time=data.get('end_time'),
            result=data.get('result'),
            error=data.get('error')
        )


@dataclass
class Plan:
    """A plan consisting of ordered actions to achieve a goal"""
    plan_id: str
    goal_id: str
    name: str
    description: str
    actions: List[Action] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_action_index: int = 0
    progress: float = 0.0
    
    def add_action(self, action: Action) -> None:
        """Add an action to the plan"""
        self.actions.append(action)
    
    def insert_action(self, index: int, action: Action) -> None:
        """Insert an action at a specific position"""
        self.actions.insert(index, action)
    
    def remove_action(self, action_id: str) -> bool:
        """Remove an action from the plan"""
        for i, action in enumerate(self.actions):
            if action.action_id == action_id:
                self.actions.pop(i)
                return True
        return False
    
    def get_current_action(self) -> Optional[Action]:
        """Get the current action being executed"""
        if 0 <= self.current_action_index < len(self.actions):
            return self.actions[self.current_action_index]
        return None
    
    def get_next_action(self) -> Optional[Action]:
        """Get the next action to execute"""
        if self.current_action_index < len(self.actions):
            return self.actions[self.current_action_index]
        return None
    
    def advance_to_next_action(self) -> bool:
        """Advance to the next action in the plan"""
        if self.current_action_index < len(self.actions):
            self.current_action_index += 1
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if the plan is complete"""
        return self.status == PlanStatus.COMPLETED
    
    def is_active(self) -> bool:
        """Check if the plan is currently active"""
        return self.status == PlanStatus.ACTIVE
    
    def calculate_progress(self) -> float:
        """Calculate the progress of the plan"""
        if not self.actions:
            return 0.0
        
        completed = sum(1 for a in self.actions if a.is_completed())
        self.progress = completed / len(self.actions)
        return self.progress
    
    def get_remaining_actions(self) -> List[Action]:
        """Get actions that haven't been completed yet"""
        return [a for a in self.actions if not a.is_completed()]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'plan_id': self.plan_id,
            'goal_id': self.goal_id,
            'name': self.name,
            'description': self.description,
            'actions': [a.to_dict() for a in self.actions],
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'current_action_index': self.current_action_index,
            'progress': self.progress
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Plan':
        plan = cls(
            plan_id=data.get('plan_id', ''),
            goal_id=data.get('goal_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            actions=[Action.from_dict(a) for a in data.get('actions', [])],
            status=PlanStatus(data.get('status', 'draft')),
            created_at=data.get('created_at', time.time()),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            current_action_index=data.get('current_action_index', 0),
            progress=data.get('progress', 0.0)
        )
        return plan


class Planner:
    """
    Core planning engine for the AGI.
    
    This engine can:
    - Create and manage goals
    - Generate plans to achieve goals
    - Execute and monitor plans
    - Adapt plans based on changing conditions
    - Repair failed plans
    """
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.plans: Dict[str, Plan] = {}
        self.actions: Dict[str, Action] = {}
        self.current_plan: Optional[Plan] = None
        self.current_goal: Optional[Goal] = None
        self.execution_history: List[str] = []
        
    def create_goal(self, description: str, goal_id: Optional[str] = None,
                    priority: float = 0.5, deadline: Optional[float] = None,
                    **kwargs) -> Goal:
        """Create a new goal"""
        goal_id = goal_id or self._generate_id(f"goal_{description}")
        
        goal = Goal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            deadline=deadline,
            **kwargs
        )
        
        self.goals[goal_id] = goal
        return goal
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID"""
        return self.goals.get(goal_id)
    
    def update_goal(self, goal_id: str, **kwargs) -> bool:
        """Update a goal"""
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            for key, value in kwargs.items():
                if hasattr(goal, key):
                    setattr(goal, key, value)
            return True
        return False
    
    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal"""
        if goal_id in self.goals:
            del self.goals[goal_id]
            return True
        return False
    
    def create_plan(self, goal_id: str, name: str = None,
                    description: str = "") -> Optional[Plan]:
        """Create a plan for a goal"""
        if goal_id not in self.goals:
            return None
        
        goal = self.goals[goal_id]
        plan_id = self._generate_id(f"plan_{goal_id}")
        
        plan = Plan(
            plan_id=plan_id,
            goal_id=goal_id,
            name=name or f"Plan for {goal.description[:30]}...",
            description=description
        )
        
        self.plans[plan_id] = plan
        goal.status = GoalStatus.ACTIVE
        
        return plan
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID"""
        return self.plans.get(plan_id)
    
    def set_current_plan(self, plan_id: str) -> bool:
        """Set the current active plan"""
        if plan_id in self.plans:
            self.current_plan = self.plans[plan_id]
            goal = self.goals.get(self.current_plan.goal_id)
            if goal:
                self.current_goal = goal
            return True
        return False
    
    def plan_for_goal(self, goal_id: str, max_actions: int = 10) -> Optional[Plan]:
        """
        Generate a plan to achieve a goal.
        
        This is a simplified version. In a full implementation, this would
        use the LLM to generate creative plans based on the goal.
        """
        if goal_id not in self.goals:
            return None
        
        goal = self.goals[goal_id]
        plan = self.create_plan(goal_id)
        
        if not plan:
            return None
        
        # Generate actions based on goal description
        actions = self._generate_actions_for_goal(goal, max_actions)
        
        for action in actions:
            plan.add_action(action)
        
        plan.status = PlanStatus.DRAFT
        return plan
    
    def _generate_actions_for_goal(self, goal: Goal, max_actions: int) -> List[Action]:
        """Generate actions to achieve a goal"""
        actions = []
        
        # Simple action generation based on goal description
        description = goal.description.lower()
        
        # Extract potential actions from the goal
        action_verbs = ['create', 'build', 'find', 'get', 'make', 'write', 
                       'read', 'analyze', 'process', 'send', 'receive', 
                       'move', 'go', 'take', 'give', 'use', 'learn']
        
        for i, verb in enumerate(action_verbs[:max_actions]):
            if verb in description:
                action = Action(
                    action_id=self._generate_id(f"action_{goal.goal_id}_{i}"),
                    name=f"{verb.capitalize()} action",
                    description=f"Perform {verb} to achieve: {goal.description[:50]}",
                    goal_id=goal.goal_id,
                    preconditions=[f"Goal {goal.goal_id} is active"],
                    postconditions=[f"Progress toward {goal.description}"],
                    estimated_duration=5.0,
                    priority=goal.priority * 0.8
                )
                actions.append(action)
        
        # If no specific actions found, create generic ones
        if not actions:
            for i in range(min(3, max_actions)):
                action = Action(
                    action_id=self._generate_id(f"action_{goal.goal_id}_{i}"),
                    name=f"Step {i+1}",
                    description=f"Step {i+1} toward: {goal.description[:50]}",
                    goal_id=goal.goal_id,
                    estimated_duration=5.0,
                    priority=goal.priority
                )
                actions.append(action)
        
        return actions
    
    def execute_plan(self, plan_id: str, action_executor: Callable) -> bool:
        """Execute a plan using the provided action executor"""
        plan = self.get_plan(plan_id)
        if not plan:
            return False
        
        if plan.status != PlanStatus.DRAFT and plan.status != PlanStatus.PAUSED:
            return False
        
        plan.status = PlanStatus.ACTIVE
        plan.started_at = time.time()
        
        for action in plan.actions:
            if not action.is_completed():
                plan.current_action_index = plan.actions.index(action)
                
                # Execute the action
                success = action.execute(action_executor)
                
                if not success:
                    # Plan failed
                    plan.status = PlanStatus.FAILED
                    return False
                
                # Record execution
                self.execution_history.append(action.action_id)
        
        # All actions completed
        plan.status = PlanStatus.COMPLETED
        plan.completed_at = time.time()
        plan.progress = 1.0
        
        # Mark goal as achieved
        goal = self.goals.get(plan.goal_id)
        if goal:
            goal.status = GoalStatus.ACHIEVED
            goal.progress = 1.0
        
        return True
    
    def execute_next_action(self, plan_id: str, action_executor: Callable) -> bool:
        """Execute just the next action in a plan"""
        plan = self.get_plan(plan_id)
        if not plan or plan.status != PlanStatus.ACTIVE:
            return False
        
        next_action = plan.get_next_action()
        if not next_action:
            # Plan is complete
            plan.status = PlanStatus.COMPLETED
            plan.completed_at = time.time()
            return True
        
        # Execute the action
        success = next_action.execute(action_executor)
        
        if success:
            plan.advance_to_next_action()
            plan.calculate_progress()
            self.execution_history.append(next_action.action_id)
            
            # If this was the last action, mark plan as complete
            if not plan.get_next_action():
                plan.status = PlanStatus.COMPLETED
                plan.completed_at = time.time()
        else:
            plan.status = PlanStatus.FAILED
        
        return success
    
    def adapt_plan(self, plan_id: str, new_information: str) -> bool:
        """
        Adapt a plan based on new information.
        
        This would use the LLM to modify the plan intelligently.
        For now, this is a placeholder.
        """
        plan = self.get_plan(plan_id)
        if not plan:
            return False
        
        # In a full implementation, we would:
        # 1. Analyze new information
        # 2. Identify which parts of the plan are affected
        # 3. Modify or replace affected actions
        # 4. Re-validate the plan
        
        # For now, just mark that adaptation occurred
        return True
    
    def repair_plan(self, plan_id: str) -> bool:
        """
        Repair a failed plan.
        
        This would identify why the plan failed and create a new plan
        or modify the existing one to avoid the failure.
        """
        plan = self.get_plan(plan_id)
        if not plan or plan.status != PlanStatus.FAILED:
            return False
        
        # In a full implementation, we would:
        # 1. Analyze the failure
        # 2. Identify the failed action
        # 3. Create alternative actions
        # 4. Generate a new plan or modify the existing one
        
        # For now, reset the plan to draft status
        plan.status = PlanStatus.DRAFT
        plan.current_action_index = 0
        
        # Reset all actions to pending
        for action in plan.actions:
            if action.is_failed():
                action.status = ActionStatus.PENDING
                action.start_time = None
                action.end_time = None
                action.error = None
        
        return True
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(str(timestamp).encode()).hexdigest()[:4]
        return f"{prefix}_{timestamp}_{random_part}"
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals"""
        return [g for g in self.goals.values() if g.is_active()]
    
    def get_pending_goals(self) -> List[Goal]:
        """Get all pending goals"""
        return [g for g in self.goals.values() if g.is_pending()]
    
    def get_achieved_goals(self) -> List[Goal]:
        """Get all achieved goals"""
        return [g for g in self.goals.values() if g.is_achieved()]
    
    def get_failed_goals(self) -> List[Goal]:
        """Get all failed goals"""
        return [g for g in self.goals.values() if g.is_failed()]
    
    def get_active_plans(self) -> List[Plan]:
        """Get all active plans"""
        return [p for p in self.plans.values() if p.is_active()]
    
    def clear(self) -> None:
        """Clear all goals, plans, and actions"""
        self.goals.clear()
        self.plans.clear()
        self.actions.clear()
        self.current_plan = None
        self.current_goal = None
        self.execution_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get planning statistics"""
        return {
            'total_goals': len(self.goals),
            'active_goals': len(self.get_active_goals()),
            'pending_goals': len(self.get_pending_goals()),
            'achieved_goals': len(self.get_achieved_goals()),
            'failed_goals': len(self.get_failed_goals()),
            'total_plans': len(self.plans),
            'active_plans': len(self.get_active_plans()),
            'completed_plans': len([p for p in self.plans.values() if p.is_complete()]),
            'total_actions_executed': len(self.execution_history)
        }
    
    def save_state(self, path: str = "planner_state.json") -> None:
        """Save planner state to file"""
        state = {
            'goals': {k: v.to_dict() for k, v in self.goals.items()},
            'plans': {k: v.to_dict() for k, v in self.plans.items()},
            'actions': {k: v.to_dict() for k, v in self.actions.items()},
            'execution_history': self.execution_history
        }
        
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str = "planner_state.json") -> bool:
        """Load planner state from file"""
        try:
            import os
            if os.path.exists(path):
                with open(path, 'r') as f:
                    state = json.load(f)
                
                self.goals = {k: Goal.from_dict(v) for k, v in state.get('goals', {}).items()}
                self.plans = {k: Plan.from_dict(v) for k, v in state.get('plans', {}).items()}
                self.actions = {k: Action.from_dict(v) for k, v in state.get('actions', {}).items()}
                self.execution_history = state.get('execution_history', [])
                
                return True
        except FileNotFoundError:
            pass
        return False

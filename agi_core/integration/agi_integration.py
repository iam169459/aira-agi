"""
AGI Integration - Unified Interface with Purple AI

This module provides a unified interface that integrates the AGI Core
with the existing Purple AI system, enabling:
- AGI-enhanced reasoning with emotion awareness
- Memory-integrated conversation
- Goal-driven planning with mood adaptation
- Self-improving capabilities
"""

import time
import json
import os
from typing import List, Dict, Optional, Any, Callable

# Import from existing Purple AI
try:
    from purple_ai.core.brain import PurpleAgent as PurpleAIAgent
    from purple_ai.mood_engine.mood_state import MoodState
    from purple_ai.safe_mode.guardrails import SafeModeGuard
    PURPLE_AVAILABLE = True
except ImportError:
    PURPLE_AVAILABLE = False
    PurpleAIAgent = None
    MoodState = None
    SafeModeGuard = None

# Import from AGI Core
from ..reasoning import ReasoningEngine, ThoughtChain, ReflectionEngine
from ..memory import MemorySystem, ShortTermMemory, LongTermMemory, EpisodicMemory
from ..planning import Planner, DecisionEngine
from ..world_model import WorldModel
from ..meta_learning import MetaLearner


class AGIIntegration:
    """
    Integration layer between AGI Core and Purple AI.
    
    This class provides a unified interface that combines:
    - Purple AI's emotion awareness and perception
    - AGI Core's reasoning, memory, planning, and meta-learning
    """
    
    def __init__(self, use_purple: bool = True):
        self.use_purple = use_purple and PURPLE_AVAILABLE
        
        # Initialize AGI Core components
        self.reasoning_engine = ReasoningEngine()
        self.memory_system = MemorySystem()
        self.planner = Planner()
        self.decision_engine = DecisionEngine()
        self.world_model = WorldModel()
        self.meta_learner = MetaLearner()
        self.reflection_engine = ReflectionEngine()
        
        # Initialize Purple AI components
        self.purple_agent = None
        self.mood_state = None
        self.safe_guard = None
        
        if self.use_purple:
            self._initialize_purple()
        
        # Integration state
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}
        self.session_start_time = time.time()
        
    def _initialize_purple(self):
        """Initialize Purple AI components"""
        if PURPLE_AVAILABLE:
            self.purple_agent = PurpleAIAgent(safe=True)
            self.mood_state = self.purple_agent.mood
            self.safe_guard = self.purple_agent.guard
    
    def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input through the integrated AGI system.
        
        This method:
        1. Analyzes the input
        2. Updates memory with new information
        3. Uses reasoning to understand intent
        4. Considers mood and emotion
        5. Generates an appropriate response
        """
        start_time = time.time()
        context = context or {}
        
        # Record the input in memory
        input_id = self.memory_system.store(
            content=user_input,
            memory_type="user_input",
            destination="stm",
            tags=["input", "conversation"]
        )
        
        # Analyze input with reasoning
        reasoning_chain = self.reasoning_engine.chain_of_thought(
            f"User input: {user_input}",
            max_steps=5
        )
        
        # Update current context
        self.current_context['last_input'] = user_input
        self.current_context['last_input_time'] = start_time
        self.current_context['reasoning_chain'] = reasoning_chain.to_dict()
        
        # Get mood if available
        mood_info = {}
        if self.mood_state:
            mood_info = {
                'emotion': self.mood_state.emotion,
                'valence': self.mood_state.valence,
                'arousal': self.mood_state.arousal,
                'dominance': self.mood_state.dominance
            }
        
        # Determine intent from reasoning
        intent = self._extract_intent(reasoning_chain)
        
        # Generate response
        response = self._generate_response(user_input, intent, mood_info, reasoning_chain)
        
        # Store response in memory
        response_id = self.memory_system.store(
            content=response,
            memory_type="response",
            destination="stm",
            tags=["response", "conversation"]
        )
        
        # Record conversation
        conversation_entry = {
            'timestamp': start_time,
            'user_input': user_input,
            'response': response,
            'intent': intent,
            'mood': mood_info,
            'reasoning': reasoning_chain.to_dict(),
            'duration': time.time() - start_time
        }
        self.conversation_history.append(conversation_entry)
        
        # Learn from the interaction
        self._learn_from_interaction(user_input, response, intent, mood_info)
        
        return {
            'response': response,
            'intent': intent,
            'mood': mood_info,
            'reasoning': reasoning_chain.to_dict(),
            'memory_updates': [input_id, response_id],
            'processing_time': time.time() - start_time
        }
    
    def _extract_intent(self, reasoning_chain: ThoughtChain) -> Dict[str, Any]:
        """Extract intent from a reasoning chain"""
        intent = {
            'type': 'unknown',
            'confidence': 0.0,
            'details': {}
        }
        
        # Simple intent extraction
        user_input = self.current_context.get('last_input', '')
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['how', 'what', 'why', 'when', 'where', 'who']):
            intent['type'] = 'question'
            intent['confidence'] = 0.8
        elif any(word in input_lower for word in ['do', 'create', 'make', 'build', 'write']):
            intent['type'] = 'request'
            intent['confidence'] = 0.8
        elif any(word in input_lower for word in ['tell', 'explain', 'describe']):
            intent['type'] = 'information_request'
            intent['confidence'] = 0.8
        elif any(word in input_lower for word in ['hello', 'hi', 'hey']):
            intent['type'] = 'greeting'
            intent['confidence'] = 0.9
        elif any(word in input_lower for word in ['bye', 'goodbye', 'exit', 'quit']):
            intent['type'] = 'farewell'
            intent['confidence'] = 0.9
        elif any(word in input_lower for word in ['thank', 'thanks']):
            intent['type'] = 'gratitude'
            intent['confidence'] = 0.9
        else:
            intent['type'] = 'statement'
            intent['confidence'] = 0.7
        
        intent['details'] = {'input': user_input}
        
        return intent
    
    def _generate_response(self, user_input: str, intent: Dict[str, Any],
                           mood_info: Dict[str, Any], 
                           reasoning_chain: ThoughtChain) -> str:
        """Generate an appropriate response"""
        # Consider mood in response generation
        emotion = mood_info.get('emotion', 'neutral')
        
        # Mood-based response modifiers
        mood_responses = {
            "stressed": "I understand this might be stressful. Let me help you with that. ",
            "angry": "I sense your frustration. Let's work through this together. ",
            "excited": "That's exciting! Tell me more about what you're thinking. ",
            "happy": "I'm glad to hear that! How can I help you today? ",
            "calm": "I appreciate your calm approach. Let's take this step by step. ",
            "neutral": "",
            "tired": "You sound tired. Let me help you with that. ",
            "sad": "I'm here for you. How can I help? "
        }
        
        mood_prefix = mood_responses.get(emotion, "")
        
        # Generate response based on intent
        intent_type = intent.get('type', 'unknown')
        
        if intent_type == 'greeting':
            base_response = "Hello! I'm TinyAGI, your AI assistant. How can I help you today?"
        elif intent_type == 'farewell':
            base_response = "Goodbye! Have a great day. I'm here if you need me."
        elif intent_type == 'gratitude':
            base_response = "You're welcome! I'm happy to help."
        elif intent_type == 'question':
            # Try to answer using knowledge
            answer = self._answer_question(user_input)
            base_response = answer if answer else "That's an interesting question. Let me think about that."
        elif intent_type == 'request':
            base_response = self._handle_request(user_input)
        elif intent_type == 'information_request':
            base_response = self._provide_information(user_input)
        else:
            base_response = "I see. That's interesting. How can I assist you?"
        
        # Combine mood prefix with base response
        response = f"{mood_prefix}{base_response}"
        
        # Add reasoning summary if confidence is high
        if reasoning_chain.confidence > 0.8:
            conclusion = reasoning_chain.get_conclusion()
            if conclusion:
                response = f"{response} [Reasoning: {conclusion[:100]}]"
        
        return response
    
    def _answer_question(self, question: str) -> Optional[str]:
        """Try to answer a question using available knowledge"""
        # Query memory for relevant information
        results = self.memory_system.query(question)
        
        if results:
            # Return the most relevant result
            best_result = results[0]
            if hasattr(best_result.item, 'content'):
                return f"Based on my knowledge: {best_result.item.content}"
            elif hasattr(best_result.item, 'definition'):
                return f"{best_result.item.name}: {best_result.item.definition}"
        
        # Query world model
        entities = self.world_model.query({'entity_type': 'any'})
        if entities:
            return f"I know about {len(entities)} entities in my world model."
        
        return None
    
    def _handle_request(self, request: str) -> str:
        """Handle a user request"""
        # Create a goal for this request
        goal = self.planner.create_goal(
            description=request,
            priority=0.8
        )
        
        # Create a plan
        plan = self.planner.plan_for_goal(goal.goal_id, max_actions=3)
        
        if plan:
            return f"I'll help you with that. I've created a plan with {len(plan.actions)} steps to achieve your goal."
        else:
            return "I'll do my best to help you with that."
    
    def _provide_information(self, request: str) -> str:
        """Provide information about a topic"""
        # Query memory
        results = self.memory_system.query(request)
        
        if results:
            info = []
            for result in results[:3]:
                if hasattr(result.item, 'content'):
                    info.append(str(result.item.content))
                elif hasattr(result.item, 'definition'):
                    info.append(f"{result.item.name}: {result.item.definition}")
            
            if info:
                return "Here's what I know: " + ". ".join(info)
        
        return "Let me share what I know about that."
    
    def _learn_from_interaction(self, user_input: str, response: str,
                               intent: Dict[str, Any], mood_info: Dict[str, Any]) -> None:
        """Learn from the user interaction"""
        # Record learning experience
        self.meta_learner.record_experience(
            task_description=f"Conversation: {intent.get('type', 'unknown')}",
            learning_method="conversation",
            duration=0.1,  # Will be updated
            success=True,  # Assume success for now
            accuracy=0.9,
            lessons=[f"Learned about {intent.get('type', 'topic')}"]
        )
        
        # Update world model with new information
        if intent.get('type') == 'information_request':
            # Extract entities from the input
            # This would be more sophisticated in a full implementation
            pass
    
    def set_mood(self, valence: float, arousal: float, dominance: float) -> None:
        """Set the current mood state"""
        if self.mood_state:
            # Update mood state
            self.mood_state.valence = valence
            self.mood_state.arousal = arousal
            self.mood_state.dominance = dominance
    
    def get_agi_state(self) -> Dict[str, Any]:
        """Get the current state of the AGI system"""
        return {
            'reasoning_engine': self.reasoning_engine.get_best_chain(),
            'memory_stats': self.memory_system.get_stats(),
            'planning_stats': self.planner.get_stats(),
            'world_model_stats': self.world_model.get_stats(),
            'meta_learning_stats': self.meta_learner.get_learning_stats(),
            'mood': {
                'emotion': self.mood_state.emotion if self.mood_state else 'unknown',
                'valence': self.mood_state.valence if self.mood_state else 0.0,
                'arousal': self.mood_state.arousal if self.mood_state else 0.0,
                'dominance': self.mood_state.dominance if self.mood_state else 0.0
            } if self.mood_state else {},
            'conversation_count': len(self.conversation_history)
        }
    
    def save_state(self, path: str = "agi_state.json") -> None:
        """Save the AGI state to file"""
        state = {
            'memory_system': self.memory_system.get_stats(),
            'planner': self.planner.get_stats(),
            'world_model': self.world_model.get_stats(),
            'meta_learner': self.meta_learner.get_learning_stats(),
            'conversation_history': self.conversation_history[-50:]  # Last 50
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str = "agi_state.json") -> bool:
        """Load the AGI state from file"""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    state = json.load(f)
                
                # Restore conversation history
                self.conversation_history = state.get('conversation_history', [])
                
                return True
        except FileNotFoundError:
            pass
        return False


class AGIAgent:
    """
    Complete AGI Agent that integrates all capabilities.
    
    This is the main class for interacting with the AGI system.
    It provides a high-level interface that combines all AGI Core
    and Purple AI capabilities.
    """
    
    def __init__(self, name: str = "TinyAGI", use_purple: bool = True):
        self.name = name
        self.integration = AGIIntegration(use_purple=use_purple)
        self.initialized = False
        
    def initialize(self) -> None:
        """Initialize the AGI agent"""
        if not self.initialized:
            # Initialize all components
            self.integration.memory_system.load_all()
            self.integration.world_model.load()
            self.integration.meta_learner.load()
            self.integration.load_state()
            
            self.initialized = True
            print(f"{self.name} initialized successfully!")
    
    def converse(self, user_input: str) -> Dict[str, Any]:
        """Have a conversation with the AGI"""
        result = self.integration.process_input(user_input)
        return result
    
    def think(self, problem: str, max_steps: int = 8) -> ThoughtChain:
        """Use the reasoning engine to think through a problem"""
        return self.integration.reasoning_engine.chain_of_thought(problem, max_steps)
    
    def remember(self, information: str, memory_type: str = "fact") -> str:
        """Store information in memory"""
        return self.integration.memory_system.store(information, memory_type)
    
    def recall(self, query: str) -> List[Any]:
        """Recall information from memory"""
        results = self.integration.memory_system.query(query)
        return [r.item for r in results]
    
    def plan(self, goal: str) -> Dict[str, Any]:
        """Create a plan to achieve a goal"""
        goal_obj = self.integration.planner.create_goal(goal)
        plan = self.integration.planner.plan_for_goal(goal_obj.goal_id)
        
        if plan:
            return {
                'goal_id': goal_obj.goal_id,
                'plan_id': plan.plan_id,
                'actions': [a.to_dict() for a in plan.actions]
            }
        return {'error': 'Failed to create plan'}
    
    def decide(self, description: str, options: List[str]) -> Dict[str, Any]:
        """Make a decision"""
        decision = self.integration.decision_engine.make_decision(
            description, options
        )
        return {
            'decision_id': decision.decision_id,
            'chosen_option': decision.chosen_option.description if decision.chosen_option else None,
            'all_options': [o.description for o in decision.options],
            'reasoning': decision.reasoning
        }
    
    def learn(self, concept: str, definition: str, examples: List[str] = None) -> str:
        """Learn a new concept"""
        return self.integration.memory_system.learn_concept(
            concept, definition, examples
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the AGI"""
        return self.integration.get_agi_state()
    
    def save(self) -> None:
        """Save the AGI state"""
        self.integration.memory_system.save_all()
        self.integration.world_model.save()
        self.integration.meta_learner.save()
        self.integration.save_state()
    
    def shutdown(self) -> None:
        """Shutdown the AGI agent"""
        self.save()
        print(f"{self.name} shutdown complete. State saved.")

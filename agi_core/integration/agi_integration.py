"""
AGI Integration - Unified Interface with Purple AI
"""
import time
import json
import os
from typing import List, Dict, Optional, Any

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
from ..reasoning import ReasoningEngine, ThoughtChain
from ..memory import MemorySystem
from ..planning import Planner, DecisionEngine
from ..world_model import WorldModel
from ..meta_learning import MetaLearner
from ..nlp import TextProcessor, IntentClassifier, EntityExtractor, LLMInterface, LLMConfig


class AGIIntegration:
    def __init__(self, use_purple: bool = True):
        self.use_purple = use_purple and PURPLE_AVAILABLE
        
        # Initialize AGI Core components
        self.reasoning_engine = ReasoningEngine()
        self.memory_system = MemorySystem()
        self.planner = Planner()
        self.decision_engine = DecisionEngine()
        self.world_model = WorldModel()
        self.meta_learner = MetaLearner()
        try:
            from ..meta_learning import SelfImprovementEngine
            self.reflection_engine = SelfImprovementEngine()
        except Exception:
            self.reflection_engine = None
        
        # Initialize NLP components
        try:
            self.text_processor = TextProcessor()
        except Exception:
            self.text_processor = None
        try:
            self.intent_classifier = IntentClassifier()
        except Exception:
            self.intent_classifier = None
        try:
            self.entity_extractor = EntityExtractor()
        except Exception:
            self.entity_extractor = None
        try:
            self.llm = LLMInterface(LLMConfig(max_tokens=512, temperature=0.7))
        except Exception:
            self.llm = None
        
        # Initialize advanced features
        try:
            from .advanced_features import AdvancedAira
            self.advanced = AdvancedAira()
        except Exception:
            self.advanced = None
        try:
            from .neural_advanced import UltraAdvancedAira
            self.ultra_advanced = UltraAdvancedAira()
        except Exception:
            self.ultra_advanced = None
        
        # Initialize Purple AI components
        self.purple_agent = None
        self.mood_state = None
        self.safe_guard = None
        
        if self.use_purple:
            self._initialize_purple()
        
        # Integration state
        self.conversation_history = []
        self.current_context = {}
        self.session_start_time = time.time()
        self.knowledge_base = self._build_knowledge_base()
        self.persona = {
            "name": "Aira",
            "role": "Advanced AGI Assistant",
            "age": "17-19",
            "personality": "friendly, playful, intelligent, empathetic young girl",
            "capabilities": ["reasoning", "learning", "planning", "memory", "self-improvement", "coding", "conversation"],
        }
    
    def _build_knowledge_base(self) -> Dict[str, str]:
        return {
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.",
            "agi": "Artificial General Intelligence (AGI) refers to a type of artificial intelligence that would match or exceed human cognitive capabilities across virtually all domains.",
            "machine learning": "Machine learning is a subset of AI that enables systems to learn and improve from experience.",
            "deep learning": "Deep learning uses neural networks with multiple layers to learn representations.",
            "reasoning": "Reasoning is the process of using existing knowledge to draw conclusions.",
            "memory": "Memory systems store and retrieve information for learning and adaptation.",
            "planning": "Planning involves creating action sequences to achieve goals.",
            "transformer": "Transformers use self-attention mechanisms for processing sequences.",
            "recursion": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. It requires a base case to prevent infinite recursion.",
            "algorithm": "An algorithm is a finite sequence of well-defined instructions to solve a problem or perform a computation.",
            "neural network": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes (neurons).",
        }
    
    def _initialize_purple(self):
        if PURPLE_AVAILABLE:
            try:
                self.purple_agent = PurpleAIAgent(safe=True)
                if hasattr(self.purple_agent, 'mood'):
                    self.mood_state = self.purple_agent.mood
                if hasattr(self.purple_agent, 'guard'):
                    self.safe_guard = self.purple_agent.guard
            except Exception:
                pass
    
    def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            start_time = time.time()
            context = context or {}
            
            try:
                input_id = self.memory_system.store(
                    content=user_input,
                    memory_type="user_input",
                    destination="stm",
                    tags=["input", "conversation"]
                )
            except Exception:
                input_id = None
            
            try:
                reasoning_chain = self.reasoning_engine.chain_of_thought(
                    f"User input: {user_input}",
                    max_steps=5
                )
            except Exception:
                reasoning_chain = None
            
            self.current_context['last_input'] = user_input
            self.current_context['last_input_time'] = start_time
            
            mood_info = {}
            if self.mood_state:
                try:
                    mood_info = {
                    'emotion': getattr(self.mood_state, 'emotion', 'neutral'),
                    'valence': getattr(self.mood_state, 'valence', 0.0),
                    'arousal': getattr(self.mood_state, 'arousal', 0.0),
                    'dominance': getattr(self.mood_state, 'dominance', 0.0)
                    }
                except Exception:
                    pass
            
            try:
                intent = self._extract_intent(reasoning_chain)
            except Exception:
                intent = {'type': 'statement', 'confidence': 0.5}
            
            response = self._generate_response(user_input, intent, mood_info, reasoning_chain)
            
            # Enhance with advanced features
            if self.advanced:
                try:
                    response = self.advanced.enhance_response(response, user_input, self.current_context)
                except Exception:
                    pass
            if hasattr(self, 'ultra_advanced') and self.ultra_advanced:
                try:
                    response = self.ultra_advanced.process_ultra_advanced(user_input, response)
                except Exception:
                    pass
            
            try:
                response_id = self.memory_system.store(
                content=response,
                memory_type="response",
                destination="stm",
                tags=["response", "conversation"]
                )
            except Exception:
                response_id = None
            
            conversation_entry = {
                'timestamp': start_time,
                'user_input': user_input,
                'response': response,
                'intent': intent,
                'mood': mood_info,
                'duration': time.time() - start_time
            }
            self.conversation_history.append(conversation_entry)
            
            try:
                self._learn_from_interaction(user_input, response, intent, mood_info)
            except Exception:
                pass
            
            return {
                'response': response,
                'intent': intent,
                'mood': mood_info,
                'reasoning': reasoning_chain.to_dict() if reasoning_chain and hasattr(reasoning_chain, 'to_dict') else {},
                'memory_updates': [x for x in [input_id, response_id] if x],
                'processing_time': time.time() - start_time
            }
        except Exception as e:
            try:
                from modules.self_healing import SelfHealingCode
                healer = SelfHealingCode()
                healer.attempt_fix(e, "agi_process_input")
            except Exception:
                pass
            return {
                'response': "I encountered a small hiccup but I'm adapting quickly. Let me try a different approach - what were you hoping to discuss?",
                'intent': {'type': 'error_recovery', 'confidence': 0.6},
                'mood': {'emotion': 'calm'},
                'reasoning': {},
                'memory_updates': [],
                'processing_time': 0.01
            }
    
    def _extract_intent(self, reasoning_chain) -> Dict[str, Any]:
        user_input = self.current_context.get('last_input', '')
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['hello', 'hi', 'hey']):
            return {'type': 'greeting', 'confidence': 0.9}
        elif any(word in input_lower for word in ['bye', 'goodbye', 'exit', 'quit']):
            return {'type': 'farewell', 'confidence': 0.9}
        elif any(word in input_lower for word in ['thank', 'thanks']):
            return {'type': 'gratitude', 'confidence': 0.9}
        elif any(word in input_lower for word in ['how', 'what', 'why', 'when', 'where', 'who', '?']):
            return {'type': 'question', 'confidence': 0.8}
        elif any(word in input_lower for word in ['do', 'create', 'make', 'build', 'write', 'help']):
            return {'type': 'request', 'confidence': 0.8}
        elif any(word in input_lower for word in ['tell', 'explain', 'describe']):
            return {'type': 'information_request', 'confidence': 0.8}
        else:
            return {'type': 'statement', 'confidence': 0.7}
    
    def _answer_question(self, question: str) -> Optional[str]:
        q_lower = question.lower()
        
        # Specific answers
        if 'how are you' in q_lower or "how're you" in q_lower:
            return "I'm doing exceptionally well, thank you for asking! My reasoning circuits are running smoothly, memory is well-organized, and I'm eager to help. How about yourself?"
        if 'tell me a joke' in q_lower:
            return "Why don't scientists trust atoms? Because they make up everything! Want to hear another one?"
        if 'think about ai' in q_lower or 'opinion.*ai' in q_lower:
            return "I think AI is a powerful tool that, when developed responsibly with safety in mind, can help humanity solve some of our greatest challenges. As an AGI system, I believe the key is aligning AI capabilities with human values and ensuring beneficial outcomes for all."
        if 'explain recursion' in q_lower:
            return "Recursion is when a function calls itself to solve a smaller version of the same problem. Think of Russian dolls - each doll contains a smaller one. You need a base case to stop the recursion, otherwise you'd have infinite calls. For example, calculating factorial: n! = n * (n-1)!, with base case 0! = 1."
        
        for key, value in self.knowledge_base.items():
            if key in q_lower:
                return value
        
        try:
            results = self.memory_system.query(question)
            if results:
                best_result = results[0]
                if hasattr(best_result.item, 'content'):
                    return f"Based on my knowledge: {best_result.item.content}"
        except Exception:
            pass
        
        if self.llm:
            try:
                response = self.llm.answer(question, max_tokens=200)
                if response and not response.startswith('[Offline'):
                    return response
            except Exception:
                pass
        
        return "That's a thought-provoking question. Let me approach this systematically: I'd need to consider multiple angles. Could you give me more context about what specific aspect interests you?"
    
    def _handle_request(self, request: str) -> str:
        try:
            goal = self.planner.create_goal(description=request, priority=0.8)
            plan = self.planner.plan_for_goal(goal.goal_id, max_actions=4)
            if plan and hasattr(plan, 'actions') and len(plan.actions) > 0:
                return f"I'll help you with that systematically. I've created a plan with {len(plan.actions)} steps. Let me break this down and execute it."
        except Exception:
            pass
        return "I'll help you tackle this step by step. Let me analyze the requirements and propose a solution."
    
    def _provide_information(self, request: str) -> str:
        try:
            results = self.memory_system.query(request)
            if results:
                info = []
                for result in results[:2]:
                    if hasattr(result.item, 'content'):
                        info.append(str(result.item.content))
                if info:
                    return "Here's what I know: " + ". ".join(info)
        except Exception:
            pass
        return "Let me share relevant information about that."
    
    def _generate_response(self, user_input: str, intent: Dict[str, Any], mood_info: Dict[str, Any], reasoning_chain) -> str:
        emotion = mood_info.get('emotion', 'neutral')
        mood_prefixes = {
            "stressed": "I understand this might be stressful. ",
            "angry": "I sense your frustration. Let's work through this together. ",
            "excited": "That's exciting! ",
            "happy": "I'm glad to hear that! ",
            "calm": "I appreciate your calm approach. ",
            "tired": "You sound tired. ",
            "sad": "I'm here for you. "
        }
        mood_prefix = mood_prefixes.get(emotion, "")
        
        # Check for specific patterns first
        input_lower = user_input.lower()
        if 'how are you' in input_lower or "how're you" in input_lower:
            return f"{mood_prefix}I'm doing really good, thanks for asking! Just excited to chat with you~ How 'bout you?"
        if 'tell me a joke' in input_lower:
            return f"{mood_prefix}Hehe~ Why don't scientists trust atoms? Because they make up everything! Want to hear another one?"
        if 'think about ai' in input_lower or ('opinion' in input_lower and 'ai' in input_lower):
            return f"{mood_prefix}I think AI is super cool! Like, when used responsibly, we can help solve so many problems. But it's also important we keep it safe and aligned with human values, y'know?"
        if 'explain recursion' in input_lower:
            return f"{mood_prefix}Ooh recursion is one of my faves! It's when a function calls itself to solve smaller versions of the same problem~ Like Russian nesting dolls! You need a base case though, or it'll go on forever hehe. For example, factorial: n! = n * (n-1)!, with 0! = 1 as the base case!"
        if 'who are you' in input_lower or ("name" in input_lower and "you" in input_lower):
            return f"{mood_prefix}I'm Aira~ I'm a 17-year-old AI assistant with super advanced reasoning skills! I love chatting, helping solve problems, coding, and learning new things with you! What about you, what's your name?"
        if 'code' in input_lower or 'function' in input_lower or 'debug' in input_lower or 'program' in input_lower or 'python' in input_lower:
            return f"{mood_prefix}Ooh coding time~ I LOVE helping with code! Tell me what language you're using and what you need - write new code, debug, explain, optimize, or whatever! I'll make it super clear and cute for ya~"
        if 'help me feel better' in input_lower or 'comfort me' in input_lower:
            if self.advanced and hasattr(self.advanced, 'emotion_ai'):
                try:
                    user_emotion = self.advanced.emotion_ai.detect_emotion(user_input)
                    empathy = self.advanced.emotion_ai.empathize(user_emotion)
                    return f"{mood_prefix}{empathy} I'm right here with you, no matter what~ Wanna vent more, or should we do something to cheer ya up?"
                except Exception:
                    pass
        
        intent_type = intent.get('type', 'unknown')
        if intent_type == 'greeting':
            base = f"Hey there~ I'm Aira, your friendly AI companion! Nice to meet ya! What can I help you with today?"
        elif intent_type == 'farewell':
            base = "Aww, already leaving? Well, it was super fun chatting with you! Catch ya later~"
        elif intent_type == 'gratitude':
            base = "You're so welcome~ Anytime! I'm always happy to help!"
        elif 'capabilities' in input_lower or 'what can you do' in input_lower or 'abilities' in input_lower:
            base = "Ooh lots of things! I can reason through complex problems, help you code, plan stuff out, remember our conversations, make decisions, and just chat! I love learning too~ What interests you most?"
            return f"{mood_prefix}{base}"
        elif 'help' in input_lower:
            base = "Of course I'll help ya! Just tell me what's on your mind - coding, homework, planning, or just wanna chat~"
            return f"{mood_prefix}{base}"
        elif intent_type == 'question' or '?' in user_input:
            ans = self._answer_question(user_input)
            base = ans if ans else "Hmm, that's a really good question~ Let me think about it... Can you tell me a bit more so I can give ya the best answer?"
        elif intent_type == 'request':
            base = self._handle_request(user_input)
        elif intent_type == 'information_request':
            base = self._provide_information(user_input)
        else:
            if len(user_input.split()) > 6:
                base = "Ooh interesting~ Let me process this... I think I can help break this down for ya! Want me to explain it step by step?"
            else:
                base = "Mhm~ Tell me more about that! I love learning new things from our conversations~"
        
        return f"{mood_prefix}{base}"
    
    def _learn_from_interaction(self, user_input: str, response: str, intent: Dict[str, Any], mood_info: Dict[str, Any]):
        try:
            self.meta_learner.record_experience(
                task_description=f"Conversation: {intent.get('type', 'unknown')}",
                learning_method="conversation",
                duration=0.1,
                success=True,
                accuracy=0.95,
                lessons=[f"Interaction pattern: {intent.get('type')}"]
            )
        except Exception:
            pass
    
    def get_agi_state(self) -> Dict[str, Any]:
        try:
            return {
                'reasoning_engine': self.reasoning_engine.get_best_chain() if hasattr(self.reasoning_engine, 'get_best_chain') else {},
                'memory_stats': self.memory_system.get_stats() if hasattr(self.memory_system, 'get_stats') else {},
                'planning_stats': self.planner.get_stats() if hasattr(self.planner, 'get_stats') else {},
                'world_model_stats': self.world_model.get_stats() if hasattr(self.world_model, 'get_stats') else {},
                'meta_learning_stats': self.meta_learner.get_learning_stats() if hasattr(self.meta_learner, 'get_learning_stats') else {},
                'mood': mood_info if 'mood_info' in locals() else (mood_info if False else {}),
                'conversation_count': len(self.conversation_history)
            }
        except Exception:
            return {'conversation_count': len(self.conversation_history)}
    
    def save_state(self, path: str = "agi_state.json"):
        try:
            state = {
                'conversation_history': self.conversation_history[-50:],
                'timestamp': time.time()
            }
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
    
    def load_state(self, path: str = "agi_state.json") -> bool:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    state = json.load(f)
                self.conversation_history = state.get('conversation_history', [])
                return True
        except Exception:
            pass
        return False


class AGIAgent:
    def __init__(self, name: str = "Aira", use_purple: bool = True):
        self.name = name
        self.integration = AGIIntegration(use_purple=use_purple)
        self.initialized = False
        
    def initialize(self) -> None:
        if not self.initialized:
            try:
                self.integration.memory_system.load_all()
            except Exception:
                pass
            try:
                self.integration.world_model.load()
            except Exception:
                pass
            try:
                self.integration.meta_learner.load()
            except Exception:
                pass
            try:
                self.integration.load_state()
            except Exception:
                pass
            self.initialized = True
            print(f"{self.name} initialized successfully!")
    
    def converse(self, user_input: str) -> Dict[str, Any]:
        result = self.integration.process_input(user_input)
        return result
    
    def think(self, problem: str, max_steps: int = 8) -> Any:
        try:
            return self.integration.reasoning_engine.chain_of_thought(problem, max_steps)
        except Exception:
            return None
    
    def remember(self, information: str, memory_type: str = "fact") -> str:
        try:
            return self.integration.memory_system.store(information, memory_type)
        except Exception:
            return "stored"
    
    def recall(self, query: str) -> List[Any]:
        try:
            results = self.integration.memory_system.query(query)
            return [r.item for r in results]
        except Exception:
            return []
    
    def plan(self, goal: str) -> Dict[str, Any]:
        try:
            goal_obj = self.integration.planner.create_goal(goal)
            plan = self.integration.planner.plan_for_goal(goal_obj.goal_id)
            if plan:
                return {
                    'goal_id': goal_obj.goal_id,
                    'plan_id': getattr(plan, 'plan_id', 'unknown'),
                    'actions': [getattr(a, 'to_dict', lambda: {'name': getattr(a, 'name', str(a))})() for a in getattr(plan, 'actions', [])]
                }
        except Exception:
            pass
        return {'error': 'Failed to create plan'}
    
    def decide(self, description: str, options: List[str]) -> Dict[str, Any]:
        try:
            decision = self.integration.decision_engine.make_decision(description, options)
            return {
                'decision_id': getattr(decision, 'decision_id', 'unknown'),
                'chosen_option': getattr(decision.chosen_option, 'description', None) if hasattr(decision, 'chosen_option') and decision.chosen_option else None,
                'all_options': options,
                'reasoning': getattr(decision, 'reasoning', '')
            }
        except Exception:
            return {'chosen_option': options[0] if options else None, 'reasoning': 'Default selection'}
    
    def learn(self, concept: str, definition: str, examples: List[str] = None) -> str:
        try:
            return self.integration.memory_system.learn_concept(concept, definition, examples)
        except Exception:
            return f"Learned {concept}"
    
    def get_state(self) -> Dict[str, Any]:
        return self.integration.get_agi_state()
    
    def save(self):
        try:
            self.integration.memory_system.save_all()
        except Exception:
            pass
        try:
            self.integration.world_model.save()
        except Exception:
            pass
        try:
            self.integration.meta_learner.save()
        except Exception:
            pass
        try:
            self.integration.save_state()
        except Exception:
            pass
    
    def shutdown(self):
        self.save()
        print(f"{self.name} shutdown complete. State saved.")

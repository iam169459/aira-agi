"""
LLM-Powered Reasoning Engine

This module provides enhanced reasoning capabilities using the transformer model.
It combines the structured reasoning of AGI Core with the generative power of
TinyGPT for more natural and capable reasoning.
"""

import sys
import os
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..reasoning import ReasoningEngine, ThoughtChain, ReasoningMode
from .transformer_integration import TransformerIntegration


@dataclass
class LLMStep:
    """A reasoning step enhanced with LLM generation"""
    step_id: str
    content: str
    llm_prompt: str
    llm_response: str
    reasoning_mode: ReasoningMode
    confidence: float = 0.8
    temperature: float = 0.7
    tokens_generated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'content': self.content,
            'llm_prompt': self.llm_prompt,
            'llm_response': self.llm_response,
            'reasoning_mode': self.reasoning_mode.value,
            'confidence': self.confidence,
            'temperature': self.temperature,
            'tokens_generated': self.tokens_generated
        }


class LLMReasoningEngine:
    """
    Enhanced reasoning engine using LLM for natural language reasoning.
    
    This engine combines:
    - Structured reasoning chains from ReasoningEngine
    - Natural language generation from TinyGPT
    - Context-aware reasoning
    """
    
    def __init__(self, model_path: str = "tiny_model.pt"):
        # Initialize base reasoning engine
        self.reasoning_engine = ReasoningEngine()
        
        # Initialize transformer integration
        self.transformer = TransformerIntegration(model_path=model_path)
        
        # Configuration
        self.default_temperature = 0.7
        self.default_max_tokens = 100
        
        # Track usage
        self.generated_tokens = 0
        self.prompts_processed = 0
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self.transformer.is_loaded()
    
    def chain_of_thought(self, prompt: str, max_steps: int = 5, 
                         temperature: float = None) -> ThoughtChain:
        """
        Perform chain-of-thought reasoning with LLM enhancement.
        
        Each step uses the LLM to generate natural language reasoning.
        """
        temperature = temperature or self.default_temperature
        chain = self.reasoning_engine.create_chain(prompt)
        
        # Generate reasoning steps using LLM
        for step_num in range(max_steps):
            # Create prompt for this step
            step_prompt = self._create_step_prompt(prompt, step_num, chain)
            
            # Generate step content using LLM
            step_content = self.transformer.generate(
                step_prompt,
                max_tokens=self.default_max_tokens,
                temperature=temperature
            )
            
            # Add step to chain
            chain.add_step(
                content=step_content,
                reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
                confidence=0.8,
                evidence=[f"LLM-generated step {step_num + 1}"]
            )
            
            # Track usage
            self.generated_tokens += min(self.default_max_tokens, len(step_content))
            self.prompts_processed += 1
        
        # Validate and calculate confidence
        chain.validate_chain()
        chain.calculate_chain_confidence()
        chain.state = chain.state.COMPLETED
        
        return chain
    
    def _create_step_prompt(self, original_prompt: str, step_num: int, 
                          chain: ThoughtChain) -> str:
        """Create a prompt for a reasoning step"""
        if step_num == 0:
            # First step: direct reasoning about the prompt
            return f"Step 1: Reason about the following: {original_prompt}\n\nThought:"
        else:
            # Subsequent steps: build on previous reasoning
            previous_content = chain.steps[-1].content if chain.steps else ""
            return f"Step {step_num + 1}: Building on previous thought: {previous_content[:100]}...\n\nNew thought about: {original_prompt[:50]}...\n\nThought:"
    
    def generate_reasoning(self, prompt: str, max_steps: int = 5,
                           temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a complete reasoning chain with LLM enhancement.
        
        Returns a dict with:
        - chain: ThoughtChain object
        - final_answer: str
        - steps: List of step contents
        - tokens_generated: int
        """
        chain = self.chain_of_thought(prompt, max_steps, temperature)
        
        return {
            'chain': chain,
            'final_answer': chain.get_conclusion() or "",
            'steps': [step.content for step in chain.steps],
            'tokens_generated': self.generated_tokens,
            'confidence': chain.confidence
        }
    
    def explain(self, concept: str, max_steps: int = 3, 
                temperature: float = 0.5) -> str:
        """Explain a concept using LLM-powered reasoning"""
        prompt = f"Explain {concept} in a clear, step-by-step manner."
        result = self.generate_reasoning(prompt, max_steps, temperature)
        return result['final_answer']
    
    def answer_question(self, question: str, max_steps: int = 4,
                        temperature: float = 0.3) -> str:
        """Answer a question using LLM-powered reasoning"""
        prompt = f"Q: {question}\n\nProvide a well-reasoned answer."
        result = self.generate_reasoning(prompt, max_steps, temperature)
        return result['final_answer']
    
    def solve_problem(self, problem: str, max_steps: int = 6,
                     temperature: float = 0.4) -> Dict[str, Any]:
        """Solve a problem using LLM-powered reasoning"""
        prompt = f"Solve the following problem: {problem}\n\nShow your reasoning step by step."
        result = self.generate_reasoning(prompt, max_steps, temperature)
        
        return {
            'problem': problem,
            'solution': result['final_answer'],
            'steps': result['steps'],
            'confidence': result.get('confidence', 0.0)
        }
    
    def analyze_text(self, text: str, max_steps: int = 4,
                     temperature: float = 0.6) -> Dict[str, Any]:
        """Analyze text using LLM-powered reasoning"""
        prompt = f"Analyze the following text: {text[:200]}...\n\nProvide insights and observations."
        result = self.generate_reasoning(prompt, max_steps, temperature)
        
        return {
            'text': text,
            'analysis': result['final_answer'],
            'steps': result['steps']
        }
    
    def compare_options(self, options: List[str], context: str = "",
                        max_steps: int = 3, temperature: float = 0.4) -> Dict[str, Any]:
        """Compare options using LLM-powered reasoning"""
        options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        prompt = f"Compare the following options in the context of: {context}\n\nOptions:\n{options_str}\n\nAnalysis:"
        
        result = self.generate_reasoning(prompt, max_steps, temperature)
        
        # Try to extract a recommendation
        final_answer = result['final_answer']
        recommendation = self._extract_recommendation(final_answer, options)
        
        return {
            'options': options,
            'context': context,
            'analysis': final_answer,
            'recommendation': recommendation,
            'steps': result['steps']
        }
    
    def _extract_recommendation(self, text: str, options: List[str]) -> Optional[str]:
        """Extract a recommendation from analysis text"""
        text_lower = text.lower()
        
        for option in options:
            option_lower = option.lower()
            # Check for positive indicators
            if any(word in text_lower for word in ['best', 'recommend', 'choose', 'suggest', 'prefer']):
                if option_lower in text_lower:
                    return option
        
        # If no clear recommendation, return the first option mentioned
        for option in options:
            if option.lower() in text_lower:
                return option
        
        return None
    
    def generate_ideas(self, topic: str, count: int = 5, 
                      max_tokens: int = 80, temperature: float = 0.8) -> List[str]:
        """Generate multiple ideas about a topic"""
        ideas = []
        
        for i in range(count):
            prompt = f"Generate idea {i+1} about: {topic}"
            idea = self.transformer.generate(prompt, max_tokens, temperature)
            ideas.append(idea.strip())
        
        return ideas
    
    def refine_answer(self, answer: str, context: str = "",
                      max_tokens: int = 100, temperature: float = 0.3) -> str:
        """Refine an answer using LLM"""
        prompt = f"Refine and improve the following answer:\n\nAnswer: {answer}\n\nContext: {context}\n\nRefined:"
        return self.transformer.generate(prompt, max_tokens, temperature).strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return self.transformer.get_model_info()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'prompts_processed': self.prompts_processed,
            'tokens_generated': self.generated_tokens,
            'model_info': self.get_model_info()
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self.generated_tokens = 0
        self.prompts_processed = 0

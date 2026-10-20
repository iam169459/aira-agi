"""
Intent Classifier - Classify user intent from text

This module provides intent classification capabilities including:
- Question detection
- Request type classification
- Sentiment detection
- Action intent extraction
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class IntentType(Enum):
    """Types of user intent"""
    UNKNOWN = "unknown"
    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION = "question"
    REQUEST = "request"
    STATEMENT = "statement"
    COMMAND = "command"
    FEEDBACK = "feedback"
    THANKS = "thanks"
    APPOLOGY = "apology"
    INFORMATION_REQUEST = "information_request"
    HELP_REQUEST = "help_request"


@dataclass
class Intent:
    """Classified intent with confidence and details"""
    intent_type: IntentType
    confidence: float = 0.0
    text: str = ""
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 (negative) to 1 (positive)
    urgency: float = 0.0  # 0 (low) to 1 (high)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'intent_type': self.intent_type.value,
            'confidence': self.confidence,
            'text': self.text,
            'entities': self.entities,
            'keywords': self.keywords,
            'sentiment': self.sentiment,
            'urgency': self.urgency
        }


class IntentClassifier:
    """
    Classifies user intent from text input.
    Uses pattern matching and keyword analysis.
    """
    
    def __init__(self):
        # Intent patterns
        self.intent_patterns = {
            IntentType.GREETING: [
                (r'\bhello\b', 1.0),
                (r'\bhi\b', 1.0),
                (r'\bhey\b', 1.0),
                (r'\bgood morning\b', 1.0),
                (r'\bgood afternoon\b', 1.0),
                (r'\bgood evening\b', 1.0),
            ],
            IntentType.FAREWELL: [
                (r'\bbye\b', 1.0),
                (r'\bgoodbye\b', 1.0),
                (r'\bsee you\b', 1.0),
                (r'\bfarewell\b', 1.0),
                (r'\btake care\b', 1.0),
                (r'\bexit\b', 0.9),
                (r'\bquit\b', 0.9),
            ],
            IntentType.QUESTION: [
                (r'\?$', 1.0),
                (r'\bwhat\b', 0.9),
                (r'\bhow\b', 0.9),
                (r'\bwhy\b', 0.9),
                (r'\bwhen\b', 0.9),
                (r'\bwhere\b', 0.9),
                (r'\bwho\b', 0.9),
                (r'\bwhich\b', 0.9),
                (r'\bcan you\b', 0.8),
                (r'\bcould you\b', 0.8),
                (r'\bwould you\b', 0.8),
            ],
            IntentType.REQUEST: [
                (r'\bplease\b', 0.8),
                (r'\bcould you\b', 0.9),
                (r'\bwould you\b', 0.9),
                (r'\bcan you\b', 0.9),
                (r'\bdo me a favor\b', 0.9),
                (r'\bhelp me\b', 0.8),
                (r'\bneed\b', 0.7),
                (r'\bwant\b', 0.7),
            ],
            IntentType.COMMAND: [
                (r'\bdo\b', 0.8),
                (r'\bcreate\b', 0.9),
                (r'\bmake\b', 0.9),
                (r'\bbuild\b', 0.9),
                (r'\bwrite\b', 0.9),
                (r'\bstart\b', 0.8),
                (r'\bstop\b', 0.8),
                (r'\bopen\b', 0.8),
                (r'\bclose\b', 0.8),
            ],
            IntentType.THANKS: [
                (r'\bthank\b', 1.0),
                (r'\bthanks\b', 1.0),
                (r'\bappreciate\b', 0.9),
                (r'\bgrateful\b', 0.9),
            ],
            IntentType.APPOLOGY: [
                (r'\bsorry\b', 1.0),
                (r'\bapologize\b', 1.0),
                (r'\bmy bad\b', 0.9),
                (r'\bforgive\b', 0.8),
            ],
            IntentType.HELP_REQUEST: [
                (r'\bhelp\b', 1.0),
                (r'\bsupport\b', 0.9),
                (r'\bassist\b', 0.8),
                (r'\bneed help\b', 0.9),
            ],
            IntentType.INFORMATION_REQUEST: [
                (r'\btell me\b', 0.9),
                (r'\bexplain\b', 0.9),
                (r'\bdescribe\b', 0.9),
                (r'\bwhat is\b', 0.9),
                (r'\bhow does\b', 0.9),
                (r'\bwhy is\b', 0.9),
            ]
        }
        
        # Sentiment words
        self.positive_words = {
            'good', 'great', 'excellent', 'wonderful', 'fantastic', 'amazing',
            'awesome', 'perfect', 'happy', 'joy', 'love', 'like', 'best',
            'success', 'win', 'beautiful', 'nice', 'positive', 'improve'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'worst', 'fail', 'failure', 'problem', 'issue', 'wrong',
            'error', 'mistake', 'negative', 'sad', 'angry', 'frustrated'
        }
        
        # Urgency words
        self.urgency_words = {
            'urgent': 1.0, 'immediately': 1.0, 'now': 0.9, 'asap': 1.0,
            'emergency': 1.0, 'critical': 0.9, 'important': 0.7,
            'hurry': 0.8, 'quick': 0.6, 'fast': 0.6
        }
    
    def classify(self, text: str) -> Intent:
        """Classify the intent of a text"""
        text_lower = text.lower()
        
        # Initialize intent
        intent = Intent(
            intent_type=IntentType.UNKNOWN,
            confidence=0.0,
            text=text,
            sentiment=self._analyze_sentiment(text_lower),
            urgency=self._analyze_urgency(text_lower)
        )
        
        # Check each intent type
        for intent_type, patterns in self.intent_patterns.items():
            for pattern, weight in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    confidence = weight
                    if confidence > intent.confidence:
                        intent.intent_type = intent_type
                        intent.confidence = confidence
        
        # Extract keywords and entities
        intent.keywords = self._extract_keywords(text_lower)
        intent.entities = self._extract_entities(text_lower)
        
        # If no intent matched with high confidence, try to infer
        if intent.confidence < 0.5:
            intent.intent_type = self._infer_intent(text_lower)
            intent.confidence = 0.5
        
        return intent
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (-1 to 1)"""
        words = text.split()
        score = 0.0
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean in self.positive_words:
                score += 0.1
            elif word_clean in self.negative_words:
                score -= 0.1
        
        # Clamp to -1 to 1
        return max(-1.0, min(1.0, score))
    
    def _analyze_urgency(self, text: str) -> float:
        """Analyze urgency of text (0 to 1)"""
        words = text.split()
        urgency = 0.0
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word).lower()
            if word_clean in self.urgency_words:
                urgency = max(urgency, self.urgency_words[word_clean])
        
        return urgency
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common words and extract nouns/verbs
        words = re.findall(r'\b\w+\b', text)
        keywords = []
        
        # Simple keyword extraction - keep words longer than 4 chars
        for word in words:
            if len(word) > 4:
                keywords.append(word)
        
        return keywords[:10]  # Return top 10
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text"""
        # Look for capitalized words (potential proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Also look for quoted phrases
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        for quote in quoted:
            if quote[0]:
                entities.append(quote[0])
            elif quote[1]:
                entities.append(quote[1])
        
        return list(set(entities))[:5]  # Return unique entities, max 5
    
    def _infer_intent(self, text: str) -> IntentType:
        """Infer intent when no pattern matches"""
        # Check if it's a question
        if '?' in text:
            return IntentType.QUESTION
        
        # Check for imperative mood (commands)
        if text.endswith('!') or any(word in text for word in ['please', 'need', 'want']):
            return IntentType.REQUEST
        
        # Default to statement
        return IntentType.STATEMENT
    
    def is_question(self, text: str) -> bool:
        """Check if text is a question"""
        intent = self.classify(text)
        return intent.intent_type == IntentType.QUESTION
    
    def get_intent_type(self, text: str) -> IntentType:
        """Get the intent type of text"""
        intent = self.classify(text)
        return intent.intent_type
    
    def get_confidence(self, text: str) -> float:
        """Get confidence score for intent classification"""
        intent = self.classify(text)
        return intent.confidence

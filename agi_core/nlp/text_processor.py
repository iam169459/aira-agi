"""
Text Processor - Preprocessing and Tokenization for NLP

This module provides text preprocessing capabilities including:
- Tokenization (word, sentence, character)
- Normalization (lowercase, stemming, lemmatization)
- Text cleaning (remove special chars, extra spaces)
- N-gram generation
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import string


@dataclass
class Token:
    """A token with position and metadata"""
    text: str
    position: int
    line: int = 0
    is_word: bool = True
    is_punctuation: bool = False
    is_number: bool = False
    
    def __str__(self):
        return self.text


class Tokenizer:
    """
    Tokenizer for splitting text into tokens.
    Supports multiple tokenization strategies.
    """
    
    def __init__(self):
        self.word_pattern = re.compile(r"\w+[-']*\w*|\w+")
        self.sentence_pattern = re.compile(r"[.!?]+|\n\n|")
        
    def tokenize_words(self, text: str) -> List[Token]:
        """Tokenize text into words"""
        tokens = []
        for i, word in enumerate(self.word_pattern.finditer(text)):
            token_text = word.group()
            token = Token(
                text=token_text,
                position=word.start(),
                is_word=True,
                is_punctuation=token_text in string.punctuation,
                is_number=token_text.isdigit()
            )
            tokens.append(token)
        return tokens
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        sentences = []
        current_sentence = []
        
        for char in text:
            current_sentence.append(char)
            if char in '.!?':
                sentences.append(''.join(current_sentence).strip())
                current_sentence = []
        
        if current_sentence:
            sentences.append(''.join(current_sentence).strip())
        
        return [s for s in sentences if s]
    
    def tokenize_characters(self, text: str) -> List[Token]:
        """Tokenize text into characters"""
        tokens = []
        for i, char in enumerate(text):
            token = Token(
                text=char,
                position=i,
                is_word=char.isalpha(),
                is_punctuation=char in string.punctuation,
                is_number=char.isdigit()
            )
            tokens.append(token)
        return tokens
    
    def tokenize(self, text: str, method: str = "word") -> List[Token]:
        """Tokenize text using specified method"""
        if method == "word":
            return self.tokenize_words(text)
        elif method == "sentence":
            sentences = self.tokenize_sentences(text)
            return [Token(text=s, position=0) for s in sentences]
        elif method == "character":
            return self.tokenize_characters(text)
        else:
            return self.tokenize_words(text)


class TextProcessor:
    """
    Text processor for cleaning and normalizing text.
    """
    
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.stop_words = self._load_stop_words()
        
    def _load_stop_words(self) -> set:
        """Load common English stop words"""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
            'don', 'should', 'now'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters and extra spaces"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters except basic punctuation
        text = re.sub(r"[^\w\s.,!?;:'\"-]", '', text)
        return text.strip()
    
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_stop_words(self, text: str) -> str:
        """Remove stop words from text"""
        tokens = self.tokenizer.tokenize_words(text)
        filtered_tokens = [t.text for t in tokens if t.text.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def stem(self, text: str) -> str:
        """Simple stemming (Porter stemmer would be better)"""
        # Simple suffix removal
        text = re.sub(r'ing$', '', text)
        text = re.sub(r'ed$', '', text)
        text = re.sub(r's$', '', text)
        text = re.sub(r'ly$', '', text)
        return text
    
    def normalize(self, text: str, remove_stop_words: bool = True, 
                  lowercase: bool = True, remove_punctuation: bool = False) -> str:
        """Normalize text with various options"""
        if lowercase:
            text = self.lowercase(text)
        
        text = self.clean_text(text)
        
        if remove_punctuation:
            text = self.remove_punctuation(text)
        
        if remove_stop_words:
            text = self.remove_stop_words(text)
        
        return text.strip()
    
    def generate_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Generate n-grams from text"""
        tokens = self.tokenizer.tokenize_words(text)
        token_texts = [t.text.lower() for t in tokens]
        
        ngrams = []
        for i in range(len(token_texts) - n + 1):
            ngram = ' '.join(token_texts[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def get_word_count(self, text: str) -> int:
        """Count words in text"""
        tokens = self.tokenizer.tokenize_words(text)
        return len([t for t in tokens if t.is_word])
    
    def get_sentence_count(self, text: str) -> int:
        """Count sentences in text"""
        sentences = self.tokenizer.tokenize_sentences(text)
        return len(sentences)
    
    def get_character_count(self, text: str) -> int:
        """Count characters in text"""
        return len(text)

"""
Entity Extractor - Extract entities from text

This module provides entity extraction capabilities including:
- Named entity recognition (simple pattern-based)
- Person, organization, location extraction
- Date and time extraction
- Number extraction
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class EntityType(Enum):
    """Types of entities"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    NUMBER = "number"
    MONEY = "money"
    PERCENTAGE = "percentage"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    """An extracted entity with type and metadata"""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'type': self.entity_type.value,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class EntityExtractor:
    """
    Extracts entities from text using pattern matching.
    """
    
    def __init__(self):
        # Patterns for different entity types
        self.patterns = {
            EntityType.PERSON: [
                # Title + Lastname (e.g., Mr. Smith, Dr. Johnson)
                (r'\b(Mr|Mrs|Ms|Dr|Prof|Rev)\s+\.?\s*[A-Z][a-z]+\b', 0.8),
                # Firstname Lastname (capitalized)
                (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', 0.7),
                # Single capitalized name
                (r'\b[A-Z][a-z]+\b', 0.5),
            ],
            EntityType.ORGANIZATION: [
                # Inc, Corp, Ltd, etc.
                (r'\b[A-Z][a-zA-Z0-9]+\s+(Inc|Corp|LLC|Ltd|LLC|Plc)\b', 0.9),
                # Company names with capital letters
                (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(Group|Company|Corporation|Systems)\b', 0.8),
                # Common organization patterns
                (r'\b(Google|Microsoft|Apple|Amazon|Facebook|Twitter|IBM|Intel|NVIDIA)\b', 1.0),
            ],
            EntityType.LOCATION: [
                # City, State
                (r'\b[A-Z][a-z]+,\s+[A-Z]{2}\b', 0.9),
                # City names
                (r'\b(New York|Los Angeles|Chicago|Houston|Phoenix|London|Paris|Tokyo|Berlin)\b', 1.0),
                # Country names
                (r'\b(United States|USA|UK|France|Germany|Japan|China|India|Canada)\b', 1.0),
                # Street addresses
                (r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b', 0.8),
            ],
            EntityType.DATE: [
                # YYYY-MM-DD
                (r'\b\d{4}-\d{2}-\d{2}\b', 1.0),
                # MM/DD/YYYY
                (r'\b\d{1,2}/\d{1,2}/\d{4}\b', 1.0),
                # DD-MM-YYYY
                (r'\b\d{1,2}-\d{1,2}-\d{4}\b', 1.0),
                # Month Day, Year
                (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', 1.0),
                # Day Month Year
                (r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', 1.0),
            ],
            EntityType.TIME: [
                # HH:MM:SS
                (r'\b\d{1,2}:\d{2}:\d{2}\b', 1.0),
                # HH:MM
                (r'\b\d{1,2}:\d{2}\b', 0.9),
                # AM/PM
                (r'\b\d{1,2}:\d{2}\s+(AM|PM|am|pm)\b', 1.0),
            ],
            EntityType.NUMBER: [
                # Integers
                (r'\b\d+\b', 0.9),
                # Decimals
                (r'\b\d+\.\d+\b', 1.0),
                # Numbers with commas
                (r'\b\d{1,3}(,\d{3})*\b', 1.0),
                # Scientific notation
                (r'\b\d+\.?\d*[eE][+-]?\d+\b', 1.0),
            ],
            EntityType.MONEY: [
                # Dollar amounts
                (r'\$\d+\.?\d*\b', 1.0),
                # Other currencies
                (r'\b(€|£|¥|₹)\d+\.?\d*\b', 1.0),
                # With commas
                (r'\$\d{1,3}(,\d{3})*\.?\d*\b', 1.0),
            ],
            EntityType.PERCENTAGE: [
                (r'\b\d+\.?\d*%\b', 1.0),
            ],
            EntityType.EMAIL: [
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 1.0),
            ],
            EntityType.URL: [
                (r'\bhttps?://[^\s]+\b', 1.0),
                (r'\bwww\.[^\s]+\b', 1.0),
            ],
            EntityType.PHONE: [
                # US phone numbers
                (r'\b\d{3}-\d{3}-\d{4}\b', 1.0),
                (r'\b\(\d{3}\)\s?\d{3}-\d{4}\b', 1.0),
                (r'\b\d{10,15}\b', 0.7),
            ],
        }
    
    def extract(self, text: str, entity_types: List[EntityType] = None) -> List[ExtractedEntity]:
        """Extract entities from text"""
        entity_types = entity_types or list(EntityType)
        entities = []
        
        for entity_type in entity_types:
            if entity_type in self.patterns:
                entities.extend(self._extract_type(text, entity_type))
        
        # Remove overlapping entities (keep the longest)
        entities = self._remove_overlaps(entities)
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        
        return entities
    
    def _extract_type(self, text: str, entity_type: EntityType) -> List[ExtractedEntity]:
        """Extract entities of a specific type"""
        entities = []
        
        for pattern, confidence in self.patterns[entity_type]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_text = match.group()
                
                # Skip if this is just a single letter
                if len(entity_text.strip()) <= 1:
                    continue
                
                # Create entity
                entity = ExtractedEntity(
                    text=entity_text,
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence
                )
                
                # Add metadata based on type
                if entity_type == EntityType.DATE:
                    entity.metadata['parsed'] = self._parse_date(entity_text)
                elif entity_type == EntityType.TIME:
                    entity.metadata['parsed'] = self._parse_time(entity_text)
                elif entity_type == EntityType.NUMBER:
                    entity.metadata['numeric'] = self._parse_number(entity_text)
                elif entity_type == EntityType.MONEY:
                    entity.metadata['amount'] = self._parse_money(entity_text)
                elif entity_type == EntityType.PERCENTAGE:
                    entity.metadata['value'] = self._parse_percentage(entity_text)
                
                entities.append(entity)
        
        return entities
    
    def _remove_overlaps(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove overlapping entities, keeping the longest"""
        # Sort by start position
        entities.sort(key=lambda e: e.start)
        
        result = []
        for entity in entities:
            # Check if this overlaps with the last entity in result
            if result and entity.start < result[-1].end:
                # Overlapping - keep the longer one
                if len(entity.text) > len(result[-1].text):
                    result[-1] = entity
            else:
                result.append(entity)
        
        return result
    
    def _parse_date(self, date_text: str) -> Optional[str]:
        """Parse date text into standard format"""
        # Simple date parsing - would use dateutil in full implementation
        return date_text
    
    def _parse_time(self, time_text: str) -> Optional[str]:
        """Parse time text into standard format"""
        return time_text
    
    def _parse_number(self, number_text: str) -> Optional[float]:
        """Parse number text into float"""
        try:
            return float(number_text.replace(',', ''))
        except ValueError:
            return None
    
    def _parse_money(self, money_text: str) -> Optional[float]:
        """Parse money text into amount"""
        try:
            # Remove currency symbols and commas
            clean = money_text.replace('$', '').replace(',', '').replace('€', '').replace('£', '')
            return float(clean)
        except ValueError:
            return None
    
    def _parse_percentage(self, percent_text: str) -> Optional[float]:
        """Parse percentage text into value (0-1)"""
        try:
            clean = percent_text.replace('%', '')
            value = float(clean) / 100.0
            return value
        except ValueError:
            return None
    
    def extract_all(self, text: str) -> List[ExtractedEntity]:
        """Extract all types of entities"""
        return self.extract(text)
    
    def extract_by_type(self, text: str, entity_type: EntityType) -> List[ExtractedEntity]:
        """Extract entities of a specific type"""
        return self.extract(text, [entity_type])
    
    def get_entity_texts(self, text: str, entity_type: EntityType = None) -> List[str]:
        """Get just the text of extracted entities"""
        entities = self.extract(text, [entity_type] if entity_type else None)
        return [e.text for e in entities]

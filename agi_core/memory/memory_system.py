"""
Memory System - Unified Interface for All Memory Types

This module provides a unified interface for the AGI's complete memory system,
integrating short-term, long-term, and episodic memory.
"""

import time
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
import json
import os

from .short_term_memory import ShortTermMemory, WorkingMemory, MemoryItem
from .long_term_memory import LongTermMemory, SemanticMemory, Concept, Fact, Procedure
from .episodic_memory import EpisodicMemory, Experience, EventSequence


@dataclass
class MemoryQuery:
    """A query to the memory system"""
    query: str
    memory_types: List[str] = field(default_factory=list)  # 'stm', 'ltm', 'episodic'
    limit: int = 10
    confidence_threshold: float = 0.0
    tags: List[str] = field(default_factory=list)
    time_range: Optional[Dict[str, float]] = None  # {'start': ..., 'end': ...}
    
    def __post_init__(self):
        if not self.memory_types:
            self.memory_types = ['stm', 'ltm', 'episodic']


@dataclass
class MemoryResult:
    """A result from a memory query"""
    memory_type: str
    item_type: str  # 'concept', 'fact', 'experience', 'memory_item'
    item: Any
    confidence: float
    relevance: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_type': self.memory_type,
            'item_type': self.item_type,
            'confidence': self.confidence,
            'relevance': self.relevance,
            'timestamp': self.timestamp
        }


class MemorySystem:
    """
    Unified memory system for the AGI.
    
    This system integrates:
    - Short-term memory (working memory, immediate context)
    - Long-term memory (semantic knowledge, concepts, facts)
    - Episodic memory (personal experiences, events)
    
    It provides a single interface for storing, retrieving, and
    reasoning about information across all memory types.
    """
    
    def __init__(self, stm_capacity: int = 50, ltm_path: str = "ltm.json",
                 episodic_path: str = "episodic.json"):
        # Initialize memory subsystems
        self.stm = ShortTermMemory(max_items=stm_capacity)
        self.working_memory = WorkingMemory(capacity=7)
        self.ltm = LongTermMemory(storage_path=ltm_path)
        self.episodic = EpisodicMemory(storage_path=episodic_path)
        
        # Memory statistics
        self.access_count = 0
        self.store_count = 0
        self.last_access_time = time.time()
        
    def store(self, content: Any, memory_type: str = "fact",
              destination: str = "auto", **kwargs) -> str:
        """
        Store information in memory.
        
        Args:
            content: The information to store
            memory_type: Type of memory item ('fact', 'concept', 'experience', etc.)
            destination: Where to store ('stm', 'ltm', 'episodic', 'auto')
            **kwargs: Additional parameters for specific memory types
        
        Returns:
            The ID of the stored item
        """
        self.store_count += 1
        
        if destination == "stm" or (destination == "auto" and self._should_use_stm(content)):
            # Store in short-term memory
            item_id = self.stm.add(content, memory_type=memory_type, **kwargs)
            return f"stm:{item_id}"
        
        elif destination == "ltm" or (destination == "auto" and memory_type in ['concept', 'fact']):
            # Store in long-term memory
            if memory_type == 'concept':
                concept = self.ltm.store_concept(
                    kwargs.get('name', str(content)[:50]),
                    kwargs.get('definition', str(content))
                )
                return f"ltm:concept:{concept.concept_id}"
            else:
                # Default to storing as a fact
                fact = self.ltm.store_fact(
                    statement=str(content),
                    subject=kwargs.get('subject', 'unknown'),
                    predicate=kwargs.get('predicate', 'has'),
                    obj=kwargs.get('object', str(content)[:50])
                )
                return f"ltm:fact:{fact.fact_id}"
        
        elif destination == "episodic" or (destination == "auto" and memory_type == 'experience'):
            # Store in episodic memory
            experience = self.episodic.add_experience(
                title=kwargs.get('title', str(content)[:50]),
                description=str(content),
                timestamp=kwargs.get('timestamp')
            )
            return f"episodic:{experience.experience_id}"
        
        else:
            # Default to short-term memory
            item_id = self.stm.add(content, memory_type=memory_type, **kwargs)
            return f"stm:{item_id}"
    
    def _should_use_stm(self, content: Any) -> bool:
        """Determine if content should be stored in short-term memory"""
        # Simple heuristic: short content or recent context goes to STM
        content_str = str(content)
        return len(content_str) < 500
    
    def retrieve(self, item_id: str) -> Optional[Any]:
        """Retrieve an item from memory by ID"""
        self.access_count += 1
        self.last_access_time = time.time()
        
        if item_id.startswith("stm:"):
            return self.stm.get(item_id[4:])
        elif item_id.startswith("ltm:concept:"):
            return self.ltm.retrieve_concept(item_id[12:])
        elif item_id.startswith("ltm:fact:"):
            return self.ltm.semantic_memory.get_fact(item_id[9:])
        elif item_id.startswith("episodic:"):
            return self.episodic.get_experience(item_id[9:])
        else:
            # Try all memory types
            # STM
            result = self.stm.get(item_id)
            if result:
                return result
            
            # LTM
            concept = self.ltm.semantic_memory.get_concept(item_id)
            if concept:
                return concept
            
            fact = self.ltm.semantic_memory.get_fact(item_id)
            if fact:
                return fact
            
            # Episodic
            return self.episodic.get_experience(item_id)
    
    def query(self, query: Union[str, MemoryQuery]) -> List[MemoryResult]:
        """
        Query memory across all types.
        
        Args:
            query: Either a string query or a MemoryQuery object
        
        Returns:
            List of MemoryResult objects
        """
        self.access_count += 1
        self.last_access_time = time.time()
        
        if isinstance(query, str):
            query = MemoryQuery(query=query)
        
        results: List[MemoryResult] = []
        
        # Query short-term memory
        if 'stm' in query.memory_types:
            stm_results = self.stm.search(query.query, query.limit)
            for item in stm_results:
                if item.confidence >= query.confidence_threshold:
                    results.append(MemoryResult(
                        memory_type='stm',
                        item_type='memory_item',
                        item=item,
                        confidence=item.confidence,
                        relevance=1.0,
                        timestamp=item.timestamp
                    ))
        
        # Query long-term memory
        if 'ltm' in query.memory_types:
            ltm_results = self.ltm.search(query.query, query.limit)
            for concept in ltm_results.get('concepts', []):
                results.append(MemoryResult(
                    memory_type='ltm',
                    item_type='concept',
                    item=concept,
                    confidence=concept.get('confidence', 1.0),
                    relevance=0.9,
                    timestamp=concept.get('timestamp', time.time())
                ))
            
            for fact in ltm_results.get('facts', []):
                results.append(MemoryResult(
                    memory_type='ltm',
                    item_type='fact',
                    item=fact,
                    confidence=fact.get('confidence', 1.0),
                    relevance=0.9,
                    timestamp=fact.get('timestamp', time.time())
                ))
        
        # Query episodic memory
        if 'episodic' in query.memory_types:
            episodic_results = self.episodic.search_experiences(query.query, query.limit)
            for experience in episodic_results:
                if query.time_range:
                    start = query.time_range.get('start', 0)
                    end = query.time_range.get('end', float('inf'))
                    if not (start <= experience.timestamp <= end):
                        continue
                
                results.append(MemoryResult(
                    memory_type='episodic',
                    item_type='experience',
                    item=experience,
                    confidence=experience.confidence,
                    relevance=0.8,
                    timestamp=experience.timestamp
                ))
        
        # Sort by relevance and confidence
        results.sort(key=lambda x: (x.relevance, x.confidence), reverse=True)
        
        return results[:query.limit]
    
    def remember_experience(self, title: str, description: str,
                           **kwargs) -> str:
        """
        Store an experience in episodic memory.
        
        This is a convenience method for storing experiences.
        """
        experience = self.episodic.add_experience(title, description, **kwargs)
        
        # Also store key information in STM for immediate access
        self.stm.add(
            content=f"Experience: {title}",
            memory_type="experience",
            tags=['experience', title.lower()]
        )
        
        return f"episodic:{experience.experience_id}"
    
    def learn_concept(self, name: str, definition: str, 
                      examples: List[str] = None) -> str:
        """
        Learn a new concept and store it in long-term memory.
        """
        concept = self.ltm.store_concept(name, definition)
        
        if examples:
            for example in examples:
                concept.add_example(example)
        
        # Store in STM for immediate use
        self.stm.add(
            content=f"Concept: {name} - {definition}",
            memory_type="concept",
            tags=['concept', name.lower()]
        )
        
        return f"ltm:concept:{concept.concept_id}"
    
    def learn_fact(self, statement: str, subject: str, predicate: str, 
                   obj: str, source: str = "") -> str:
        """
        Learn a new fact and store it in long-term memory.
        """
        fact = self.ltm.store_fact(statement, subject, predicate, obj, source)
        
        # Store in STM for immediate use
        self.stm.add(
            content=f"Fact: {statement}",
            memory_type="fact",
            tags=['fact', subject.lower(), predicate.lower()]
        )
        
        return f"ltm:fact:{fact.fact_id}"
    
    def forget(self, item_id: str) -> bool:
        """Remove an item from memory"""
        if item_id.startswith("stm:"):
            return self.stm.remove(item_id[4:])
        elif item_id.startswith("ltm:concept:"):
            concept_id = item_id[12:]
            if concept_id in self.ltm.semantic_memory.concepts:
                del self.ltm.semantic_memory.concepts[concept_id]
                return True
        elif item_id.startswith("ltm:fact:"):
            fact_id = item_id[9:]
            if fact_id in self.ltm.semantic_memory.facts:
                del self.ltm.semantic_memory.facts[fact_id]
                return True
        elif item_id.startswith("episodic:"):
            exp_id = item_id[9:]
            if exp_id in self.episodic.experiences:
                del self.episodic.experiences[exp_id]
                return True
        return False
    
    def consolidate_memory(self) -> Dict[str, Any]:
        """
        Consolidate information from short-term to long-term memory.
        
        This moves important or frequently accessed information from
        STM to LTM for permanent storage.
        """
        results = {
            'consolidated_items': 0,
            'stm_to_ltm': 0,
            'stm_to_episodic': 0
        }
        
        # Get frequently accessed STM items
        stm_items = self.stm.get_recent(20)
        
        for item in stm_items:
            # If accessed many times or has high importance tags
            if item.access_count > 3 or any(tag in ['important', 'remember'] for tag in item.tags):
                # Convert to LTM based on type
                if item.memory_type == 'experience':
                    # Store as experience in episodic memory
                    self.episodic.add_experience(
                        title=str(item.content)[:50],
                        description=str(item.content),
                        timestamp=item.timestamp
                    )
                    results['stm_to_episodic'] += 1
                else:
                    # Store as concept or fact in LTM
                    content_str = str(item.content)
                    if len(content_str) > 100:
                        # Store as concept
                        self.ltm.store_concept(
                            name=content_str[:50],
                            definition=content_str
                        )
                    else:
                        # Store as fact
                        self.ltm.store_fact(
                            statement=content_str,
                            subject="unknown",
                            predicate="has",
                            obj=content_str
                        )
                    results['stm_to_ltm'] += 1
                
                results['consolidated_items'] += 1
        
        return results
    
    def get_context(self, focus: str, n: int = 10) -> Dict[str, Any]:
        """
        Get relevant context for a given focus.
        
        This retrieves information from all memory types that is
        relevant to the current focus of attention.
        """
        # Create a context window in STM
        context_window = self.stm.create_context_window(focus=focus)
        
        # Query all memory types
        results = self.query(MemoryQuery(
            query=focus,
            limit=n
        ))
        
        # Add results to context window
        for result in results:
            self.stm.add(
                content=result.item,
                memory_type=result.item_type,
                tags=[focus, 'context']
            )
        
        return {
            'context_window_id': context_window.window_id,
            'results': results,
            'stm_context': self.stm.get_recent(5)
        }
    
    def save_all(self) -> None:
        """Save all memory to persistent storage"""
        self.ltm.save()
        self.episodic.save()
    
    def load_all(self) -> None:
        """Load all memory from persistent storage"""
        self.ltm.load()
        self.episodic.load()
    
    def clear_all(self) -> None:
        """Clear all memory"""
        self.stm.clear()
        self.working_memory.clear()
        self.ltm.semantic_memory.clear()
        self.episodic.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        return {
            'access_count': self.access_count,
            'store_count': self.store_count,
            'stm': self.stm.get_stats(),
            'ltm': self.ltm.get_stats(),
            'episodic': self.episodic.get_stats(),
            'working_memory': {
                'items': len(self.working_memory.items),
                'capacity': self.working_memory.capacity
            }
        }
    
    def set_working_memory_focus(self, focus: str) -> None:
        """Set the current focus of working memory"""
        self.working_memory.push_focus(focus)
    
    def get_working_memory_contents(self) -> List[Any]:
        """Get the current contents of working memory"""
        return self.working_memory.get_contents()

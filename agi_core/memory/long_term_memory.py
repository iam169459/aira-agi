"""
Long-Term Memory - Semantic and Structural Knowledge Storage

This module implements the AGI's long-term memory system, which stores
knowledge, concepts, and relationships in a structured format.
"""

import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict
import hashlib


@dataclass
class Concept:
    """A concept stored in semantic memory"""
    concept_id: str
    name: str
    definition: str
    examples: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    related_concepts: List[str] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    def add_example(self, example: str) -> None:
        """Add an example of this concept"""
        if example not in self.examples:
            self.examples.append(example)
    
    def add_property(self, name: str, value: Any) -> None:
        """Add a property to this concept"""
        self.properties[name] = value
    
    def add_related_concept(self, concept_id: str) -> None:
        """Add a related concept"""
        if concept_id not in self.related_concepts:
            self.related_concepts.append(concept_id)
    
    def access(self) -> None:
        """Record an access to this concept"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept_id': self.concept_id,
            'name': self.name,
            'definition': self.definition,
            'examples': self.examples,
            'properties': self.properties,
            'related_concepts': self.related_concepts,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Concept':
        return cls(
            concept_id=data.get('concept_id', ''),
            name=data.get('name', ''),
            definition=data.get('definition', ''),
            examples=data.get('examples', []),
            properties=data.get('properties', {}),
            related_concepts=data.get('related_concepts', []),
            confidence=data.get('confidence', 1.0),
            timestamp=data.get('timestamp', time.time()),
            last_accessed=data.get('last_accessed', time.time()),
            access_count=data.get('access_count', 0)
        )


@dataclass
class Fact:
    """A factual statement stored in memory"""
    fact_id: str
    statement: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    def access(self) -> None:
        """Record an access to this fact"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fact_id': self.fact_id,
            'statement': self.statement,
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence,
            'source': self.source,
            'timestamp': self.timestamp,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        return cls(
            fact_id=data.get('fact_id', ''),
            statement=data.get('statement', ''),
            subject=data.get('subject', ''),
            predicate=data.get('predicate', ''),
            object=data.get('object', ''),
            confidence=data.get('confidence', 1.0),
            source=data.get('source', ''),
            timestamp=data.get('timestamp', time.time()),
            last_accessed=data.get('last_accessed', time.time()),
            access_count=data.get('access_count', 0)
        )


@dataclass
class Procedure:
    """A procedure or skill stored in memory"""
    procedure_id: str
    name: str
    description: str
    steps: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    success_count: int = 0
    failure_count: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def add_step(self, step: str) -> None:
        """Add a step to the procedure"""
        self.steps.append(step)
    
    def record_success(self) -> None:
        """Record a successful execution"""
        self.success_count += 1
    
    def record_failure(self) -> None:
        """Record a failed execution"""
        self.failure_count += 1
    
    def get_success_rate(self) -> float:
        """Get the success rate of this procedure"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'procedure_id': self.procedure_id,
            'name': self.name,
            'description': self.description,
            'steps': self.steps,
            'parameters': self.parameters,
            'preconditions': self.preconditions,
            'postconditions': self.postconditions,
            'confidence': self.confidence,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Procedure':
        return cls(
            procedure_id=data.get('procedure_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            steps=data.get('steps', []),
            parameters=data.get('parameters', {}),
            preconditions=data.get('preconditions', []),
            postconditions=data.get('postconditions', []),
            confidence=data.get('confidence', 1.0),
            success_count=data.get('success_count', 0),
            failure_count=data.get('failure_count', 0),
            timestamp=data.get('timestamp', time.time())
        )


class SemanticMemory:
    """
    Semantic memory - stores concepts, facts, and relationships.
    
    Characteristics:
    - Structured knowledge storage
    - Concepts with definitions and examples
    - Facts as subject-predicate-object triples
    - Procedures and skills
    - Persistent storage
    """
    
    def __init__(self, storage_path: str = "semantic_memory.json"):
        self.storage_path = storage_path
        self.concepts: Dict[str, Concept] = {}
        self.facts: Dict[str, Fact] = {}
        self.procedures: Dict[str, Procedure] = {}
        
        # Indexes for fast lookup
        self.concept_name_index: Dict[str, str] = {}
        self.fact_index: Dict[Tuple[str, str, str], str] = {}
        self.procedure_name_index: Dict[str, str] = {}
        
        self.load()
        
    def add_concept(self, name: str, definition: str, 
                    concept_id: Optional[str] = None) -> Concept:
        """Add a concept to semantic memory"""
        concept_id = concept_id or self._generate_id(f"concept_{name}")
        
        concept = Concept(
            concept_id=concept_id,
            name=name,
            definition=definition
        )
        
        self.concepts[concept_id] = concept
        self.concept_name_index[name.lower()] = concept_id
        
        return concept
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID"""
        return self.concepts.get(concept_id)
    
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get a concept by name"""
        concept_id = self.concept_name_index.get(name.lower())
        if concept_id:
            return self.concepts.get(concept_id)
        return None
    
    def add_fact(self, statement: str, subject: str, predicate: str, 
                 obj: str, source: str = "", fact_id: Optional[str] = None) -> Fact:
        """Add a fact to semantic memory"""
        fact_id = fact_id or self._generate_id(f"fact_{subject}_{predicate}_{obj}")
        
        fact = Fact(
            fact_id=fact_id,
            statement=statement,
            subject=subject,
            predicate=predicate,
            object=obj,
            source=source
        )
        
        self.facts[fact_id] = fact
        self.fact_index[(subject, predicate, obj)] = fact_id
        
        return fact
    
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID"""
        return self.facts.get(fact_id)
    
    def get_facts_about(self, subject: str) -> List[Fact]:
        """Get all facts about a subject"""
        results = []
        for fact in self.facts.values():
            if fact.subject == subject:
                results.append(fact)
        return results
    
    def add_procedure(self, name: str, description: str,
                      procedure_id: Optional[str] = None) -> Procedure:
        """Add a procedure to semantic memory"""
        procedure_id = procedure_id or self._generate_id(f"proc_{name}")
        
        procedure = Procedure(
            procedure_id=procedure_id,
            name=name,
            description=description
        )
        
        self.procedures[procedure_id] = procedure
        self.procedure_name_index[name.lower()] = procedure_id
        
        return procedure
    
    def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        """Get a procedure by ID"""
        return self.procedures.get(procedure_id)
    
    def get_procedure_by_name(self, name: str) -> Optional[Procedure]:
        """Get a procedure by name"""
        procedure_id = self.procedure_name_index.get(name.lower())
        if procedure_id:
            return self.procedures.get(procedure_id)
        return None
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """Search concepts by name or definition"""
        query_lower = query.lower()
        results = []
        
        for concept in self.concepts.values():
            if (query_lower in concept.name.lower() or 
                query_lower in concept.definition.lower()):
                results.append(concept)
        
        results.sort(key=lambda x: x.access_count, reverse=True)
        return results[:limit]
    
    def search_facts(self, query: str, limit: int = 10) -> List[Fact]:
        """Search facts by content"""
        query_lower = query.lower()
        results = []
        
        for fact in self.facts.values():
            if (query_lower in fact.statement.lower() or
                query_lower in fact.subject.lower() or
                query_lower in fact.object.lower()):
                results.append(fact)
        
        results.sort(key=lambda x: x.access_count, reverse=True)
        return results[:limit]
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(str(timestamp).encode()).hexdigest()[:4]
        return f"{prefix}_{timestamp}_{random_part}"
    
    def save(self) -> None:
        """Save semantic memory to file"""
        state = {
            'concepts': {k: v.to_dict() for k, v in self.concepts.items()},
            'facts': {k: v.to_dict() for k, v in self.facts.items()},
            'procedures': {k: v.to_dict() for k, v in self.procedures.items()}
        }
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self) -> bool:
        """Load semantic memory from file"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    state = json.load(f)
                
                self.concepts = {k: Concept.from_dict(v) for k, v in state.get('concepts', {}).items()}
                self.facts = {k: Fact.from_dict(v) for k, v in state.get('facts', {}).items()}
                self.procedures = {k: Procedure.from_dict(v) for k, v in state.get('procedures', {}).items()}
                
                # Rebuild indexes
                for concept_id, concept in self.concepts.items():
                    self.concept_name_index[concept.name.lower()] = concept_id
                
                for fact_id, fact in self.facts.items():
                    self.fact_index[(fact.subject, fact.predicate, fact.object)] = fact_id
                
                for proc_id, proc in self.procedures.items():
                    self.procedure_name_index[proc.name.lower()] = proc_id
                
                return True
        except FileNotFoundError:
            pass
        return False
    
    def clear(self) -> None:
        """Clear all semantic memory"""
        self.concepts.clear()
        self.facts.clear()
        self.procedures.clear()
        self.concept_name_index.clear()
        self.fact_index.clear()
        self.procedure_name_index.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'total_concepts': len(self.concepts),
            'total_facts': len(self.facts),
            'total_procedures': len(self.procedures),
            'most_accessed_concept': max(self.concepts.values(), key=lambda x: x.access_count).name if self.concepts else None,
            'most_accessed_fact': max(self.facts.values(), key=lambda x: x.access_count).statement if self.facts else None
        }


class LongTermMemory:
    """
    Long-term memory system combining semantic and episodic memory.
    
    This is the main interface for long-term memory operations.
    """
    
    def __init__(self, storage_path: str = "semantic_memory.json"):
        self.semantic_memory = SemanticMemory(storage_path=storage_path)
        # Episodic memory is handled separately
        
    def store_concept(self, name: str, definition: str) -> Concept:
        """Store a concept in long-term memory"""
        return self.semantic_memory.add_concept(name, definition)
    
    def retrieve_concept(self, name: str) -> Optional[Concept]:
        """Retrieve a concept from long-term memory"""
        return self.semantic_memory.get_concept_by_name(name)
    
    def store_fact(self, statement: str, subject: str, predicate: str, 
                   obj: str, source: str = "") -> Fact:
        """Store a fact in long-term memory"""
        return self.semantic_memory.add_fact(statement, subject, predicate, obj, source)
    
    def store_procedure(self, name: str, description: str) -> Procedure:
        """Store a procedure in long-term memory"""
        return self.semantic_memory.add_procedure(name, description)
    
    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search long-term memory"""
        return {
            'concepts': [c.to_dict() for c in self.semantic_memory.search_concepts(query, limit)],
            'facts': [f.to_dict() for f in self.semantic_memory.search_facts(query, limit)]
        }
    
    def save(self) -> None:
        """Save long-term memory"""
        self.semantic_memory.save()
    
    def load(self) -> None:
        """Load long-term memory"""
        self.semantic_memory.load()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self.semantic_memory.get_stats()

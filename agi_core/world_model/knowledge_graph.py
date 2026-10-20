"""
Knowledge Graph - Structured Knowledge Representation

This module implements a knowledge graph for representing entities,
relationships, and properties in a structured format that enables
semantic reasoning and inference.
"""

import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict
import hashlib


@dataclass
class Property:
    """A property of an entity"""
    name: str
    value: Any
    data_type: str = "string"  # string, number, boolean, list, etc.
    confidence: float = 1.0
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'data_type': self.data_type,
            'confidence': self.confidence,
            'source': self.source,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Property':
        return cls(
            name=data.get('name', ''),
            value=data.get('value'),
            data_type=data.get('data_type', 'string'),
            confidence=data.get('confidence', 1.0),
            source=data.get('source', ''),
            timestamp=data.get('timestamp', time.time())
        )


@dataclass
class Entity:
    """An entity in the knowledge graph"""
    entity_id: str
    name: str
    entity_type: str = "thing"  # thing, person, place, concept, event, etc.
    description: str = ""
    properties: Dict[str, Property] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    confidence: float = 1.0
    
    def add_property(self, name: str, value: Any, data_type: str = "string",
                     confidence: float = 1.0, source: str = "") -> Property:
        """Add a property to this entity"""
        prop = Property(
            name=name,
            value=value,
            data_type=data_type,
            confidence=confidence,
            source=source
        )
        self.properties[name] = prop
        self.updated_at = time.time()
        return prop
    
    def get_property(self, name: str) -> Optional[Property]:
        """Get a property by name"""
        return self.properties.get(name)
    
    def remove_property(self, name: str) -> bool:
        """Remove a property"""
        if name in self.properties:
            del self.properties[name]
            self.updated_at = time.time()
            return True
        return False
    
    def has_property(self, name: str, value: Any = None) -> bool:
        """Check if entity has a property with optional value"""
        if name not in self.properties:
            return False
        if value is not None:
            return self.properties[name].value == value
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'name': self.name,
            'entity_type': self.entity_type,
            'description': self.description,
            'properties': {k: v.to_dict() for k, v in self.properties.items()},
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        entity = cls(
            entity_id=data.get('entity_id', ''),
            name=data.get('name', ''),
            entity_type=data.get('entity_type', 'thing'),
            description=data.get('description', ''),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            confidence=data.get('confidence', 1.0)
        )
        entity.properties = {k: Property.from_dict(v) for k, v in data.get('properties', {}).items()}
        return entity


@dataclass
class Relationship:
    """A relationship between entities in the knowledge graph"""
    relationship_id: str
    relationship_type: str  # e.g., "is_a", "has", "part_of", "causes", "located_at"
    source_entity_id: str
    target_entity_id: str
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    properties: Dict[str, Property] = field(default_factory=dict)
    is_directional: bool = True
    
    def add_property(self, name: str, value: Any, data_type: str = "string",
                     confidence: float = 1.0, source: str = "") -> Property:
        """Add a property to this relationship"""
        prop = Property(
            name=name,
            value=value,
            data_type=data_type,
            confidence=confidence,
            source=source
        )
        self.properties[name] = prop
        return prop
    
    def get_inverse_relationship_type(self) -> str:
        """Get the inverse relationship type"""
        inverse_map = {
            "is_a": "is_a",  # is_a is symmetric in some contexts
            "has": "has",
            "part_of": "has_part",
            "has_part": "part_of",
            "causes": "caused_by",
            "caused_by": "causes",
            "located_at": "contains",
            "contains": "located_at",
            "connected_to": "connected_to",
            "related_to": "related_to",
            "parent_of": "child_of",
            "child_of": "parent_of",
            "friend_of": "friend_of",
            "knows": "known_by",
            "known_by": "knows",
        }
        return inverse_map.get(self.relationship_type, self.relationship_type)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'relationship_id': self.relationship_id,
            'relationship_type': self.relationship_type,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'properties': {k: v.to_dict() for k, v in self.properties.items()},
            'is_directional': self.is_directional
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        rel = cls(
            relationship_id=data.get('relationship_id', ''),
            relationship_type=data.get('relationship_type', ''),
            source_entity_id=data.get('source_entity_id', ''),
            target_entity_id=data.get('target_entity_id', ''),
            confidence=data.get('confidence', 1.0),
            created_at=data.get('created_at', time.time()),
            is_directional=data.get('is_directional', True)
        )
        rel.properties = {k: Property.from_dict(v) for k, v in data.get('properties', {}).items()}
        return rel


class KnowledgeGraph:
    """
    Knowledge graph for structured knowledge representation.
    
    This graph stores:
    - Entities with properties
    - Relationships between entities
    - Type hierarchies
    - Semantic constraints
    """
    
    def __init__(self, storage_path: str = "knowledge_graph.json"):
        self.storage_path = storage_path
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # Indexes for fast lookup
        self.entity_name_index: Dict[str, str] = {}
        self.entity_type_index: Dict[str, Set[str]] = defaultdict(set)
        self.relationship_index: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
        self.incoming_relationships: Dict[str, Set[str]] = defaultdict(set)
        self.outgoing_relationships: Dict[str, Set[str]] = defaultdict(set)
        
        self.load()
        
    def add_entity(self, name: str, entity_type: str = "thing",
                   entity_id: Optional[str] = None, description: str = "") -> Entity:
        """Add an entity to the knowledge graph"""
        entity_id = entity_id or self._generate_id(f"entity_{name}")
        
        entity = Entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            description=description
        )
        
        self.entities[entity_id] = entity
        self.entity_name_index[name.lower()] = entity_id
        self.entity_type_index[entity_type].add(entity_id)
        
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get an entity by name"""
        entity_id = self.entity_name_index.get(name.lower())
        if entity_id:
            return self.entities.get(entity_id)
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        entity_ids = self.entity_type_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def add_relationship(self, relationship_type: str, source_entity_id: str,
                        target_entity_id: str, relationship_id: Optional[str] = None,
                        is_directional: bool = True) -> Relationship:
        """Add a relationship between entities"""
        relationship_id = relationship_id or self._generate_id(
            f"rel_{relationship_type}_{source_entity_id}_{target_entity_id}"
        )
        
        rel = Relationship(
            relationship_id=relationship_id,
            relationship_type=relationship_type,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            is_directional=is_directional
        )
        
        self.relationships[relationship_id] = rel
        
        # Update indexes
        self.relationship_index[(source_entity_id, relationship_type, target_entity_id)].add(relationship_id)
        self.outgoing_relationships[source_entity_id].add(relationship_id)
        self.incoming_relationships[target_entity_id].add(relationship_id)
        
        # If not directional, also add inverse
        if not is_directional:
            inverse_type = rel.get_inverse_relationship_type()
            self.relationship_index[(target_entity_id, inverse_type, source_entity_id)].add(relationship_id)
        
        return rel
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID"""
        return self.relationships.get(relationship_id)
    
    def get_relationships(self, entity_id: str, relationship_type: str = None,
                         direction: str = "both") -> List[Relationship]:
        """Get relationships for an entity"""
        results = []
        
        if direction in ["outgoing", "both"]:
            for rel_id in self.outgoing_relationships.get(entity_id, set()):
                rel = self.relationships.get(rel_id)
                if rel and (relationship_type is None or rel.relationship_type == relationship_type):
                    results.append(rel)
        
        if direction in ["incoming", "both"]:
            for rel_id in self.incoming_relationships.get(entity_id, set()):
                rel = self.relationships.get(rel_id)
                if rel and (relationship_type is None or rel.relationship_type == relationship_type):
                    results.append(rel)
        
        return results
    
    def get_related_entities(self, entity_id: str, relationship_type: str = None,
                            direction: str = "both") -> List[Entity]:
        """Get entities related to a given entity"""
        relationships = self.get_relationships(entity_id, relationship_type, direction)
        
        entity_ids = set()
        for rel in relationships:
            if rel.source_entity_id != entity_id:
                entity_ids.add(rel.source_entity_id)
            if rel.target_entity_id != entity_id:
                entity_ids.add(rel.target_entity_id)
        
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def query(self, query: Dict[str, Any]) -> List[Entity]:
        """
        Query the knowledge graph.
        
        Query format:
        {
            'entity_type': 'person',
            'properties': {'age': {'gt': 30}},
            'relationships': [
                {'type': 'works_at', 'target': 'company_x'}
            ]
        }
        """
        results = []
        
        # Filter by entity type
        entity_type = query.get('entity_type')
        if entity_type:
            candidates = self.get_entities_by_type(entity_type)
        else:
            candidates = list(self.entities.values())
        
        # Filter by properties
        properties = query.get('properties', {})
        for entity in candidates:
            match = True
            for prop_name, prop_query in properties.items():
                prop = entity.get_property(prop_name)
                if not prop:
                    match = False
                    break
                
                # Check property value against query
                if not self._check_property_value(prop.value, prop_query):
                    match = False
                    break
            
            if match:
                results.append(entity)
        
        # Filter by relationships
        relationships = query.get('relationships', [])
        for rel_query in relationships:
            filtered_results = []
            for entity in results:
                rels = self.get_relationships(
                    entity.entity_id,
                    rel_query.get('type'),
                    rel_query.get('direction', 'outgoing')
                )
                
                # Check if any relationship matches
                target_id = rel_query.get('target')
                if target_id:
                    has_match = any(
                        (rel.source_entity_id == target_id or rel.target_entity_id == target_id)
                        for rel in rels
                    )
                else:
                    has_match = len(rels) > 0
                
                if has_match:
                    filtered_results.append(entity)
            
            results = filtered_results
        
        return results
    
    def _check_property_value(self, value: Any, query: Dict[str, Any]) -> bool:
        """Check if a property value matches a query"""
        for op, op_value in query.items():
            if op == 'eq':
                if value != op_value:
                    return False
            elif op == 'gt':
                if not (value > op_value):
                    return False
            elif op == 'lt':
                if not (value < op_value):
                    return False
            elif op == 'gte':
                if not (value >= op_value):
                    return False
            elif op == 'lte':
                if not (value <= op_value):
                    return False
            elif op == 'contains':
                if isinstance(value, str) and op_value not in value:
                    return False
                elif isinstance(value, list) and op_value not in value:
                    return False
            elif op == 'in':
                if value not in op_value:
                    return False
        return True
    
    def infer_type_hierarchy(self, entity_id: str) -> List[str]:
        """Infer the type hierarchy for an entity"""
        types = []
        entity = self.get_entity(entity_id)
        if not entity:
            return types
        
        # Start with the entity's own type
        current_type = entity.entity_type
        types.append(current_type)
        
        # Follow "is_a" relationships to find parent types
        is_a_rels = self.get_relationships(entity_id, "is_a", "outgoing")
        for rel in is_a_rels:
            parent_entity = self.get_entity(rel.target_entity_id)
            if parent_entity:
                parent_type = parent_entity.entity_type
                if parent_type not in types:
                    types.append(parent_type)
                    # Recursively get parent types
                    parent_types = self.infer_type_hierarchy(rel.target_entity_id)
                    for pt in parent_types:
                        if pt not in types:
                            types.append(pt)
        
        return types
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(str(timestamp).encode()).hexdigest()[:4]
        return f"{prefix}_{timestamp}_{random_part}"
    
    def save(self) -> None:
        """Save knowledge graph to file"""
        state = {
            'entities': {k: v.to_dict() for k, v in self.entities.items()},
            'relationships': {k: v.to_dict() for k, v in self.relationships.items()}
        }
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self) -> bool:
        """Load knowledge graph from file"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    state = json.load(f)
                
                self.entities = {k: Entity.from_dict(v) for k, v in state.get('entities', {}).items()}
                self.relationships = {k: Relationship.from_dict(v) for k, v in state.get('relationships', {}).items()}
                
                # Rebuild indexes
                for entity_id, entity in self.entities.items():
                    self.entity_name_index[entity.name.lower()] = entity_id
                    self.entity_type_index[entity.entity_type].add(entity_id)
                
                for rel_id, rel in self.relationships.items():
                    self.relationship_index[(rel.source_entity_id, rel.relationship_type, rel.target_entity_id)].add(rel_id)
                    self.outgoing_relationships[rel.source_entity_id].add(rel_id)
                    self.incoming_relationships[rel.target_entity_id].add(rel_id)
                
                return True
        except FileNotFoundError:
            pass
        return False
    
    def clear(self) -> None:
        """Clear the knowledge graph"""
        self.entities.clear()
        self.relationships.clear()
        self.entity_name_index.clear()
        self.entity_type_index.clear()
        self.relationship_index.clear()
        self.incoming_relationships.clear()
        self.outgoing_relationships.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'entity_types': {k: len(v) for k, v in self.entity_type_index.items()},
            'most_common_entity_type': max(self.entity_type_index.items(), key=lambda x: len(x[1]))[0] if self.entity_type_index else None,
            'relationship_types': self._count_relationship_types()
        }
    
    def _count_relationship_types(self) -> Dict[str, int]:
        """Count the number of each relationship type"""
        counts = defaultdict(int)
        for rel in self.relationships.values():
            counts[rel.relationship_type] += 1
        return dict(counts)

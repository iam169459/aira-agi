"""
World Model - Integrated Model of the World

This module provides a comprehensive world model that integrates:
- Knowledge graph (semantic relationships)
- Spatial model (locations and spatial relationships)
- Temporal model (events and timelines)
- Causal model (cause and effect)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
import json
import os

from .knowledge_graph import KnowledgeGraph, Entity, Relationship


@dataclass
class WorldState:
    """The current state of the world as understood by the AGI"""
    timestamp: float = field(default_factory=time.time)
    known_entities: Set[str] = field(default_factory=set)
    known_relationships: Set[str] = field(default_factory=set)
    current_location: Optional[str] = None
    current_time: float = field(default_factory=time.time)
    active_entities: Set[str] = field(default_factory=set)
    recent_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'known_entities': list(self.known_entities),
            'known_relationships': list(self.known_relationships),
            'current_location': self.current_location,
            'current_time': self.current_time,
            'active_entities': list(self.active_entities),
            'recent_changes': self.recent_changes
        }


class WorldModel:
    """
    Comprehensive world model for the AGI.
    
    This model integrates multiple aspects of world knowledge:
    - Semantic knowledge (entities, types, relationships)
    - Spatial knowledge (locations, distances, paths)
    - Temporal knowledge (events, sequences, timelines)
    - Causal knowledge (cause-effect relationships)
    
    The world model provides a unified interface for:
    - Querying world state
    - Updating world knowledge
    - Making inferences about the world
    - Simulating world changes
    """
    
    def __init__(self):
        # Initialize sub-models
        self.knowledge_graph = KnowledgeGraph()
        
        # Additional models would be initialized here:
        # self.spatial_model = SpatialModel()
        # self.temporal_model = TemporalModel()
        # self.causal_model = CausalModel()
        
        # World state
        self.world_state = WorldState()
        self.change_history: List[Dict[str, Any]] = []
        
    def add_entity(self, name: str, entity_type: str = "thing",
                   description: str = "", **properties) -> Entity:
        """Add an entity to the world model"""
        entity = self.knowledge_graph.add_entity(name, entity_type, description)
        
        # Add properties
        for prop_name, prop_value in properties.items():
            entity.add_property(prop_name, prop_value)
        
        # Update world state
        self.world_state.known_entities.add(entity.entity_id)
        self._record_change("add_entity", {"entity_id": entity.entity_id, "name": name})
        
        return entity
    
    def add_relationship(self, relationship_type: str, source: str, target: str,
                        **properties) -> Relationship:
        """Add a relationship between entities"""
        rel = self.knowledge_graph.add_relationship(
            relationship_type, source, target
        )
        
        # Add properties to relationship
        for prop_name, prop_value in properties.items():
            rel.add_property(prop_name, prop_value)
        
        # Update world state
        self.world_state.known_relationships.add(rel.relationship_id)
        self._record_change("add_relationship", {
            "relationship_id": rel.relationship_id,
            "type": relationship_type,
            "source": source,
            "target": target
        })
        
        return rel
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity from the world model"""
        return self.knowledge_graph.get_entity(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        return self.knowledge_graph.get_entities_by_type(entity_type)
    
    def query(self, query: Dict[str, Any]) -> List[Entity]:
        """Query the world model"""
        return self.knowledge_graph.query(query)
    
    def infer(self, entity_id: str, inference_type: str = "type_hierarchy") -> Any:
        """Make inferences about entities in the world"""
        if inference_type == "type_hierarchy":
            return self.knowledge_graph.infer_type_hierarchy(entity_id)
        elif inference_type == "related_entities":
            return self.knowledge_graph.get_related_entities(entity_id)
        else:
            return None
    
    def update_entity(self, entity_id: str, **updates) -> bool:
        """Update an entity in the world model"""
        entity = self.get_entity(entity_id)
        if not entity:
            return False
        
        old_values = {}
        for key, value in updates.items():
            if hasattr(entity, key):
                old_values[key] = getattr(entity, key)
                setattr(entity, key, value)
            elif key in entity.properties:
                old_values[key] = entity.properties[key].value
                entity.properties[key].value = value
        
        # Record the change
        self._record_change("update_entity", {
            "entity_id": entity_id,
            "updates": updates,
            "old_values": old_values
        })
        
        return True
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from the world model"""
        entity = self.get_entity(entity_id)
        if not entity:
            return False
        
        # Record the removal
        self._record_change("remove_entity", {
            "entity_id": entity_id,
            "name": entity.name
        })
        
        # Remove from knowledge graph
        del self.knowledge_graph.entities[entity_id]
        
        # Update world state
        self.world_state.known_entities.discard(entity_id)
        
        return True
    
    def _record_change(self, change_type: str, details: Dict[str, Any]) -> None:
        """Record a change to the world model"""
        change = {
            "timestamp": time.time(),
            "type": change_type,
            "details": details
        }
        self.change_history.append(change)
        self.world_state.recent_changes.append(change)
        
        # Keep only recent changes
        if len(self.world_state.recent_changes) > 100:
            self.world_state.recent_changes = self.world_state.recent_changes[-100:]
        
        if len(self.change_history) > 1000:
            self.change_history = self.change_history[-1000:]
    
    def get_world_state(self) -> WorldState:
        """Get the current world state"""
        return self.world_state
    
    def update_world_state(self, **updates) -> None:
        """Update the world state"""
        for key, value in updates.items():
            if hasattr(self.world_state, key):
                setattr(self.world_state, key, value)
    
    def simulate_change(self, change_description: str) -> Dict[str, Any]:
        """
        Simulate a change to the world and predict its effects.
        
        This would use the causal model to predict what would happen
        if the described change occurred.
        """
        # In a full implementation, this would:
        # 1. Parse the change description
        # 2. Identify affected entities and relationships
        # 3. Use causal model to predict effects
        # 4. Return predicted new world state
        
        return {
            'change': change_description,
            'predicted_effects': [],
            'confidence': 0.5,
            'message': 'Simulation not yet implemented'
        }
    
    def check_consistency(self) -> Dict[str, Any]:
        """Check the consistency of the world model"""
        issues = []
        
        # Check for orphaned relationships
        for rel_id, rel in self.knowledge_graph.relationships.items():
            if rel.source_entity_id not in self.knowledge_graph.entities:
                issues.append(f"Orphaned relationship {rel_id}: source entity missing")
            if rel.target_entity_id not in self.knowledge_graph.entities:
                issues.append(f"Orphaned relationship {rel_id}: target entity missing")
        
        # Check for duplicate entity names
        name_count = defaultdict(int)
        for entity in self.knowledge_graph.entities.values():
            name_count[entity.name.lower()] += 1
        
        for name, count in name_count.items():
            if count > 1:
                issues.append(f"Duplicate entity name: {name} ({count} entities)")
        
        return {
            'is_consistent': len(issues) == 0,
            'issues': issues,
            'entity_count': len(self.knowledge_graph.entities),
            'relationship_count': len(self.knowledge_graph.relationships)
        }
    
    def save(self, path: str = "world_model.json") -> None:
        """Save the world model to file"""
        state = {
            'world_state': self.world_state.to_dict(),
            'knowledge_graph': {
                'entities': {k: v.to_dict() for k, v in self.knowledge_graph.entities.items()},
                'relationships': {k: v.to_dict() for k, v in self.knowledge_graph.relationships.items()}
            },
            'change_history': self.change_history
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: str = "world_model.json") -> bool:
        """Load the world model from file"""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    state = json.load(f)
                
                # Load world state
                ws_data = state.get('world_state', {})
                self.world_state = WorldState(
                    timestamp=ws_data.get('timestamp', time.time()),
                    known_entities=set(ws_data.get('known_entities', [])),
                    known_relationships=set(ws_data.get('known_relationships', [])),
                    current_location=ws_data.get('current_location'),
                    current_time=ws_data.get('current_time', time.time()),
                    active_entities=set(ws_data.get('active_entities', [])),
                    recent_changes=ws_data.get('recent_changes', [])
                )
                
                # Load knowledge graph
                kg_data = state.get('knowledge_graph', {})
                self.knowledge_graph.entities = {
                    k: Entity.from_dict(v) for k, v in kg_data.get('entities', {}).items()
                }
                self.knowledge_graph.relationships = {
                    k: Relationship.from_dict(v) for k, v in kg_data.get('relationships', {}).items()
                }
                
                # Rebuild indexes
                for entity_id, entity in self.knowledge_graph.entities.items():
                    self.knowledge_graph.entity_name_index[entity.name.lower()] = entity_id
                    self.knowledge_graph.entity_type_index[entity.entity_type].add(entity_id)
                
                for rel_id, rel in self.knowledge_graph.relationships.items():
                    self.knowledge_graph.relationship_index[(
                        rel.source_entity_id, rel.relationship_type, rel.target_entity_id
                    )].add(rel_id)
                    self.knowledge_graph.outgoing_relationships[rel.source_entity_id].add(rel_id)
                    self.knowledge_graph.incoming_relationships[rel.target_entity_id].add(rel_id)
                
                # Load change history
                self.change_history = state.get('change_history', [])
                
                return True
        except FileNotFoundError:
            pass
        return False
    
    def clear(self) -> None:
        """Clear the world model"""
        self.knowledge_graph.clear()
        self.world_state = WorldState()
        self.change_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get world model statistics"""
        return {
            'entities': len(self.knowledge_graph.entities),
            'relationships': len(self.knowledge_graph.relationships),
            'entity_types': {k: len(v) for k, v in self.knowledge_graph.entity_type_index.items()},
            'known_entities': len(self.world_state.known_entities),
            'known_relationships': len(self.world_state.known_relationships),
            'recent_changes': len(self.world_state.recent_changes)
        }

"""
Episodic Memory - Experience and Event Storage

This module implements the AGI's episodic memory system, which stores
personal experiences, events, and temporal sequences.
"""

import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import hashlib


@dataclass
class Experience:
    """A single experience or event in episodic memory"""
    experience_id: str
    title: str
    description: str
    timestamp: float
    duration: float = 0.0  # Duration in seconds
    location: str = ""
    participants: List[str] = field(default_factory=list)
    emotions: Dict[str, float] = field(default_factory=dict)  # emotion -> intensity (0-1)
    tags: List[str] = field(default_factory=list)
    related_experiences: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0-1
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def access(self) -> None:
        """Record an access to this experience"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def add_participant(self, participant: str) -> None:
        """Add a participant to this experience"""
        if participant not in self.participants:
            self.participants.append(participant)
    
    def add_emotion(self, emotion: str, intensity: float) -> None:
        """Add or update an emotion for this experience"""
        self.emotions[emotion] = max(0.0, min(1.0, intensity))
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to this experience"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def add_related_experience(self, experience_id: str) -> None:
        """Add a related experience"""
        if experience_id not in self.related_experiences:
            self.related_experiences.append(experience_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experience_id': self.experience_id,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'location': self.location,
            'participants': self.participants,
            'emotions': self.emotions,
            'tags': self.tags,
            'related_experiences': self.related_experiences,
            'importance': self.importance,
            'confidence': self.confidence,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        return cls(
            experience_id=data.get('experience_id', ''),
            title=data.get('title', ''),
            description=data.get('description', ''),
            timestamp=data.get('timestamp', time.time()),
            duration=data.get('duration', 0.0),
            location=data.get('location', ''),
            participants=data.get('participants', []),
            emotions=data.get('emotions', {}),
            tags=data.get('tags', []),
            related_experiences=data.get('related_experiences', []),
            importance=data.get('importance', 0.5),
            confidence=data.get('confidence', 1.0),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', time.time())
        )


@dataclass
class EventSequence:
    """A sequence of related events"""
    sequence_id: str
    name: str
    events: List[str] = field(default_factory=list)  # List of experience_ids
    start_time: float = 0.0
    end_time: float = 0.0
    
    def add_event(self, experience_id: str) -> None:
        """Add an experience to the sequence"""
        if experience_id not in self.events:
            self.events.append(experience_id)
    
    def get_duration(self) -> float:
        """Get the duration of the sequence"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sequence_id': self.sequence_id,
            'name': self.name,
            'events': self.events,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class EpisodicMemory:
    """
    Episodic memory - stores personal experiences and events.
    
    Characteristics:
    - Temporal organization
    - Personal experiences
    - Emotional context
    - Spatial context
    - Event sequences
    """
    
    def __init__(self, storage_path: str = "episodic_memory.json"):
        self.storage_path = storage_path
        self.experiences: Dict[str, Experience] = {}
        self.sequences: Dict[str, EventSequence] = {}
        
        # Indexes for fast lookup
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.time_index: Dict[float, str] = {}
        self.participant_index: Dict[str, List[str]] = defaultdict(list)
        self.location_index: Dict[str, List[str]] = defaultdict(list)
        
        self.load()
        
    def add_experience(self, title: str, description: str,
                       experience_id: Optional[str] = None,
                       timestamp: Optional[float] = None) -> Experience:
        """Add an experience to episodic memory"""
        timestamp = timestamp or time.time()
        experience_id = experience_id or self._generate_id(f"exp_{title}")
        
        experience = Experience(
            experience_id=experience_id,
            title=title,
            description=description,
            timestamp=timestamp
        )
        
        self.experiences[experience_id] = experience
        self.time_index[timestamp] = experience_id
        
        return experience
    
    def get_experience(self, experience_id: str) -> Optional[Experience]:
        """Get an experience by ID"""
        return self.experiences.get(experience_id)
    
    def update_experience(self, experience_id: str, **kwargs) -> bool:
        """Update an experience"""
        if experience_id in self.experiences:
            experience = self.experiences[experience_id]
            for key, value in kwargs.items():
                if hasattr(experience, key):
                    setattr(experience, key, value)
            return True
        return False
    
    def add_experience_to_sequence(self, sequence_name: str, 
                                   experience_id: str) -> EventSequence:
        """Add an experience to a sequence"""
        # Find or create sequence
        sequence_id = self._generate_id(f"seq_{sequence_name}")
        
        if sequence_id not in self.sequences:
            self.sequences[sequence_id] = EventSequence(
                sequence_id=sequence_id,
                name=sequence_name
            )
        
        sequence = self.sequences[sequence_id]
        sequence.add_event(experience_id)
        
        # Update sequence timestamps
        experience = self.experiences.get(experience_id)
        if experience:
            if sequence.start_time == 0 or experience.timestamp < sequence.start_time:
                sequence.start_time = experience.timestamp
            if experience.timestamp > sequence.end_time:
                sequence.end_time = experience.timestamp
        
        return sequence
    
    def get_experiences_by_tag(self, tag: str) -> List[Experience]:
        """Get experiences with a specific tag"""
        experience_ids = self.tag_index.get(tag, [])
        return [self.experiences.get(eid) for eid in experience_ids if eid in self.experiences]
    
    def get_experiences_by_participant(self, participant: str) -> List[Experience]:
        """Get experiences with a specific participant"""
        experience_ids = self.participant_index.get(participant.lower(), [])
        return [self.experiences.get(eid) for eid in experience_ids if eid in self.experiences]
    
    def get_experiences_by_location(self, location: str) -> List[Experience]:
        """Get experiences at a specific location"""
        experience_ids = self.location_index.get(location.lower(), [])
        return [self.experiences.get(eid) for eid in experience_ids if eid in self.experiences]
    
    def get_experiences_in_time_range(self, start: float, end: float) -> List[Experience]:
        """Get experiences within a time range"""
        results = []
        for experience in self.experiences.values():
            if start <= experience.timestamp <= end:
                results.append(experience)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results
    
    def get_recent_experiences(self, n: int = 10) -> List[Experience]:
        """Get most recent experiences"""
        sorted_experiences = sorted(self.experiences.values(), 
                                     key=lambda x: x.timestamp, 
                                     reverse=True)
        return sorted_experiences[:n]
    
    def get_important_experiences(self, n: int = 10) -> List[Experience]:
        """Get most important experiences"""
        sorted_experiences = sorted(self.experiences.values(),
                                     key=lambda x: (x.importance, x.access_count),
                                     reverse=True)
        return sorted_experiences[:n]
    
    def search_experiences(self, query: str, limit: int = 10) -> List[Experience]:
        """Search experiences by content"""
        query_lower = query.lower()
        results = []
        
        for experience in self.experiences.values():
            if (query_lower in experience.title.lower() or
                query_lower in experience.description.lower()):
                results.append(experience)
        
        results.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
        return results[:limit]
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(str(timestamp).encode()).hexdigest()[:4]
        return f"{prefix}_{timestamp}_{random_part}"
    
    def _update_indexes(self, experience: Experience) -> None:
        """Update indexes for an experience"""
        for tag in experience.tags:
            if experience.experience_id not in self.tag_index[tag]:
                self.tag_index[tag].append(experience.experience_id)
        
        for participant in experience.participants:
            if experience.experience_id not in self.participant_index[participant.lower()]:
                self.participant_index[participant.lower()].append(experience.experience_id)
        
        if experience.location:
            if experience.experience_id not in self.location_index[experience.location.lower()]:
                self.location_index[experience.location.lower()].append(experience.experience_id)
    
    def save(self) -> None:
        """Save episodic memory to file"""
        state = {
            'experiences': {k: v.to_dict() for k, v in self.experiences.items()},
            'sequences': {k: v.to_dict() for k, v in self.sequences.items()}
        }
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self) -> bool:
        """Load episodic memory from file"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    state = json.load(f)
                
                self.experiences = {k: Experience.from_dict(v) for k, v in state.get('experiences', {}).items()}
                self.sequences = {k: EventSequence.from_dict(v) for k, v in state.get('sequences', {}).items()}
                
                # Rebuild indexes
                for experience in self.experiences.values():
                    self.time_index[experience.timestamp] = experience.experience_id
                    self._update_indexes(experience)
                
                return True
        except FileNotFoundError:
            pass
        return False
    
    def clear(self) -> None:
        """Clear all episodic memory"""
        self.experiences.clear()
        self.sequences.clear()
        self.tag_index.clear()
        self.time_index.clear()
        self.participant_index.clear()
        self.location_index.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'total_experiences': len(self.experiences),
            'total_sequences': len(self.sequences),
            'total_tags': len(self.tag_index),
            'total_participants': len(self.participant_index),
            'total_locations': len(self.location_index),
            'most_accessed': max(self.experiences.values(), key=lambda x: x.access_count).title if self.experiences else None
        }

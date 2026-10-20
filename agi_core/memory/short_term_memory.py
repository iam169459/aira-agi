"""
Short-Term Memory - Working Memory and Immediate Context

This module implements the AGI's short-term memory system, which holds
information currently being processed and provides fast access to recent context.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from collections import OrderedDict
import json


@dataclass
class MemoryItem:
    """A single item in short-term memory"""
    content: Any
    memory_type: str = "fact"  # fact, observation, hypothesis, question, intention
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    
    def access(self) -> None:
        """Record an access to this memory item"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def is_recent(self, threshold_seconds: float = 300) -> bool:
        """Check if this item was accessed recently"""
        return time.time() - self.last_accessed < threshold_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'memory_type': self.memory_type,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        return cls(
            content=data.get('content'),
            memory_type=data.get('memory_type', 'fact'),
            confidence=data.get('confidence', 1.0),
            timestamp=data.get('timestamp', time.time()),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', time.time()),
            tags=data.get('tags', [])
        )


@dataclass
class ContextWindow:
    """A contextual window of related memory items"""
    window_id: str
    items: List[MemoryItem] = field(default_factory=list)
    focus: str = ""  # Current focus of attention
    recency_score: float = 1.0
    relevance_score: float = 1.0
    
    def add_item(self, item: MemoryItem) -> None:
        """Add an item to the context window"""
        self.items.append(item)
    
    def get_recent_items(self, n: int = 5) -> List[MemoryItem]:
        """Get the most recently accessed items"""
        return sorted(self.items, key=lambda x: x.last_accessed, reverse=True)[:n]
    
    def get_by_tag(self, tag: str) -> List[MemoryItem]:
        """Get items with a specific tag"""
        return [item for item in self.items if tag in item.tags]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'window_id': self.window_id,
            'items': [item.to_dict() for item in self.items],
            'focus': self.focus,
            'recency_score': self.recency_score,
            'relevance_score': self.relevance_score
        }


class ShortTermMemory:
    """
    Short-term memory system with limited capacity.
    
    Characteristics:
    - Limited capacity (typically 7 +/- 2 items for humans, more for AGI)
    - Fast access
    - Decay over time
    - Context-based organization
    """
    
    def __init__(self, max_items: int = 50, decay_rate: float = 0.01,
                 max_context_windows: int = 5):
        self.max_items = max_items
        self.decay_rate = decay_rate
        self.max_context_windows = max_context_windows
        
        self.items: OrderedDict[str, MemoryItem] = OrderedDict()
        self.context_windows: Dict[str, ContextWindow] = {}
        self.current_window: Optional[ContextWindow] = None
        self.access_history: List[str] = []
        
    def add(self, content: Any, memory_type: str = "fact",
            confidence: float = 1.0, tags: List[str] = None) -> str:
        """Add an item to short-term memory"""
        # Generate a unique ID
        item_id = f"stm_{int(time.time() * 1000)}_{len(self.items)}"
        
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            confidence=confidence,
            tags=tags or []
        )
        
        # If at capacity, remove oldest
        while len(self.items) >= self.max_items:
            self.items.popitem(last=False)
        
        self.items[item_id] = item
        
        # Add to current context window if exists
        if self.current_window:
            self.current_window.add_item(item)
        
        return item_id
    
    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get an item from memory"""
        if item_id in self.items:
            item = self.items[item_id]
            item.access()
            self.access_history.append(item_id)
            return item
        return None
    
    def remove(self, item_id: str) -> bool:
        """Remove an item from memory"""
        if item_id in self.items:
            del self.items[item_id]
            return True
        return False
    
    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search memory for items matching query"""
        results = []
        query_lower = query.lower()
        
        for item_id, item in self.items.items():
            # Simple string matching
            content_str = str(item.content).lower()
            if query_lower in content_str:
                results.append(item)
            
            # Check tags
            for tag in item.tags:
                if query_lower in tag.lower():
                    results.append(item)
                    break
        
        # Sort by recency and relevance
        results.sort(key=lambda x: (x.last_accessed, x.access_count), reverse=True)
        return results[:limit]
    
    def create_context_window(self, window_id: str = None, 
                              focus: str = "") -> ContextWindow:
        """Create a new context window"""
        window_id = window_id or f"ctx_{int(time.time() * 1000)}"
        
        # Clean up old windows if at capacity
        if len(self.context_windows) >= self.max_context_windows:
            # Remove oldest window
            oldest_id = min(self.context_windows.keys(), 
                          key=lambda x: min([item.timestamp for item in self.context_windows[x].items]) if self.context_windows[x].items else float('inf'))
            del self.context_windows[oldest_id]
        
        window = ContextWindow(window_id=window_id, focus=focus)
        self.context_windows[window_id] = window
        self.current_window = window
        return window
    
    def set_current_window(self, window_id: str) -> bool:
        """Set the current active context window"""
        if window_id in self.context_windows:
            self.current_window = self.context_windows[window_id]
            return True
        return False
    
    def decay_memory(self) -> None:
        """Apply decay to memory items"""
        current_time = time.time()
        items_to_remove = []
        
        for item_id, item in self.items.items():
            # Decay confidence over time
            age = current_time - item.timestamp
            decay_factor = 1.0 - (self.decay_rate * age)
            item.confidence = max(0.1, item.confidence * decay_factor)
            
            # Remove if confidence is too low or too old
            if item.confidence < 0.2:
                items_to_remove.append(item_id)
            elif age > 3600:  # 1 hour
                items_to_remove.append(item_id)
        
        for item_id in items_to_remove:
            self.remove(item_id)
    
    def get_recent(self, n: int = 10) -> List[MemoryItem]:
        """Get most recently accessed items"""
        return sorted(self.items.values(), 
                     key=lambda x: x.last_accessed, 
                     reverse=True)[:n]
    
    def get_by_type(self, memory_type: str) -> List[MemoryItem]:
        """Get items of a specific type"""
        return [item for item in self.items.values() 
                if item.memory_type == memory_type]
    
    def get_by_tag(self, tag: str) -> List[MemoryItem]:
        """Get items with a specific tag"""
        return [item for item in self.items.values() if tag in item.tags]
    
    def clear(self) -> None:
        """Clear all short-term memory"""
        self.items.clear()
        self.context_windows.clear()
        self.current_window = None
        self.access_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'total_items': len(self.items),
            'max_capacity': self.max_items,
            'context_windows': len(self.context_windows),
            'access_history_length': len(self.access_history),
            'memory_types': {item.memory_type: len([i for i in self.items.values() if i.memory_type == item.memory_type]) 
                           for item in self.items.values()}
        }


class WorkingMemory:
    """
    Working memory - a specialized form of short-term memory
    that holds information currently being manipulated.
    
    This is the "workspace" where active thinking happens.
    """
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[MemoryItem] = []
        self.focus_stack: List[str] = []  # Stack of current foci
        
    def add(self, content: Any, memory_type: str = "working",
            confidence: float = 1.0) -> bool:
        """Add an item to working memory"""
        if len(self.items) >= self.capacity:
            # Remove least recently used
            self.items.pop(0)
        
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            confidence=confidence
        )
        self.items.append(item)
        return True
    
    def push_focus(self, focus: str) -> None:
        """Push a new focus onto the stack"""
        self.focus_stack.append(focus)
    
    def pop_focus(self) -> Optional[str]:
        """Pop the current focus from the stack"""
        if self.focus_stack:
            return self.focus_stack.pop()
        return None
    
    def get_current_focus(self) -> Optional[str]:
        """Get the current focus"""
        if self.focus_stack:
            return self.focus_stack[-1]
        return None
    
    def get_item(self, index: int) -> Optional[MemoryItem]:
        """Get item by index"""
        if 0 <= index < len(self.items):
            return self.items[index]
        return None
    
    def clear(self) -> None:
        """Clear working memory"""
        self.items.clear()
        self.focus_stack.clear()
    
    def get_contents(self) -> List[Any]:
        """Get all contents in working memory"""
        return [item.content for item in self.items]

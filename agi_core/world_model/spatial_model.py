"""
Spatial Model - Locations and spatial relationships
"""
from dataclasses import dataclass


@dataclass
class Location:
    name: str


@dataclass
class SpatialRelationship:
    source: str
    target: str
    relationship_type: str


class SpatialModel:
    pass

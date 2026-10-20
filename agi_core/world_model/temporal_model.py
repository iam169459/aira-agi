"""
Temporal Model - Events and timelines
"""
from dataclasses import dataclass


@dataclass
class Event:
    name: str


@dataclass
class Timeline:
    name: str


class TemporalModel:
    pass

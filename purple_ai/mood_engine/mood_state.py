"""Persistent mood engine: valence, arousal, dominance with decay and persistence."""
import json
import os
from typing import Any, Dict, List

EMOTION_CATEGORIES = ["stressed", "angry", "excited", "happy", "calm", "neutral", "tired", "sad"]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class MoodState:
    """Persistent mood with valence / arousal / dominance dimensions."""

    def __init__(self, valence=0.5, arousal=0.5, dominance=0.5, decay_rate=0.92, history=None):
        self.valence = float(valence)
        self.arousal = float(arousal)
        self.dominance = float(dominance)
        self.decay_rate = float(decay_rate)
        self.history: List[Dict[str, Any]] = list(history or [])

    @property
    def emotion(self) -> str:
        idx = int((_clamp(self.arousal, 0, 1) * 3 + _clamp(self.valence, -1, 1) * 2 + self.dominance * 2)) % 8
        return EMOTION_CATEGORIES[idx]

    def update(self, new_mood: Dict[str, Any]) -> str:
        """Blend the new observation in, pulling the mood toward neutral over time."""
        incoming_v = new_mood.get("valence")
        incoming_a = new_mood.get("arousal")
        incoming_d = new_mood.get("dominance")

        if incoming_v is not None:
            self.valence = self.valence * self.decay_rate + float(incoming_v) * (1 - self.decay_rate)
        if incoming_a is not None:
            self.arousal = _clamp(float(incoming_a), 0, 1) * self.decay_rate + 0.5 * (1 - self.decay_rate)
        if incoming_d is not None:
            self.dominance = _clamp(float(incoming_d), 0, 1) * self.decay_rate + 0.5 * (1 - self.decay_rate)

        self.history.append({
            "emotion": self.emotion,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        })
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return self.emotion

    def decay(self, steps: int = 1) -> None:
        """Move the mood back toward neutral without new input."""
        for _ in range(steps):
            self.valence *= self.decay_rate
            self.arousal = self.arousal * self.decay_rate + 0.5 * (1 - self.decay_rate)
            self.dominance = self.dominance * self.decay_rate + 0.5 * (1 - self.decay_rate)

    def describe(self) -> str:
        if len(self.history) > 1:
            trend = "descending" if self.history[-1]["valence"] < self.history[-2]["valence"] else "rising"
        else:
            trend = "stable"
        return f"{self.emotion} ({trend}, V={self.valence:.2f}, A={self.arousal:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "decay_rate": self.decay_rate,
            "emotion_history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MoodState":
        return cls(
            valence=data.get("valence", 0.5),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
            decay_rate=data.get("decay_rate", 0.92),
            history=data.get("emotion_history", []),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "MoodState":
        if not os.path.exists(path):
            return cls()
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

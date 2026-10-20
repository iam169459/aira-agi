"""Emotionaware perception: emotion detection from audio + screen capture."""
import os

class EmotionPerception:
    """Microphone-based emotion detection wrapper."""

    CATEGORIES = ["stressed","angry","excited","happy","calm","neutral","tired","sad"]

    def __init__(self):
        self.current = {"emotion":"neutral","valence":0.0,"arousal":0.5,"dominance":0.5}

    def detect(self, audio_path: str = None) -> dict:
        # In a real deployment this connects to modules/emotion EmotionFeatureExtractor
        if audio_path and os.path.exists(audio_path):
            # Placeholder: in production use librosa + PyTorch model
            pass
        return self.current

    def update_from_feature(self, emotion: str, valence: float, arousal: float, dominance: float):
        self.current = {"emotion": emotion, "valence": valence, "arousal": arousal, "dominance": dominance}

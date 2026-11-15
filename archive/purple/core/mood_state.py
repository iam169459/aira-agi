import time
import json
import os

class MoodState:
    """
    Persistent mood engine with Valence, Arousal, and Dominance (VAD) dimensions.
    """
    STATE_FILE = "data/mood_state.json"

    def __init__(self):
        self.valence = 0.0    # -1.0 (Sad/Unpleasant) to 1.0 (Happy/Pleasant)
        self.arousal = 0.0    # -1.0 (Calm/Sleepy) to 1.0 (Excited/Alert)
        self.dominance = 0.0  # -1.0 (Submissive/Fearful) to 1.0 (Influential/Bold)
        
        self.last_update = time.time()
        self.decay_rate = 0.05  # Amount mood returns to 0.0 per minute
        
        self.load_state()

    def update_mood(self, delta_v=0, delta_a=0, delta_d=0):
        self.apply_decay()
        self.valence = max(-1.0, min(1.0, self.valence + delta_v))
        self.arousal = max(-1.0, min(1.0, self.arousal + delta_a))
        self.dominance = max(-1.0, min(1.0, self.dominance + delta_d))
        self.last_update = time.time()
        self.save_state()

    def apply_decay(self):
        now = time.time()
        elapsed_minutes = (now - self.last_update) / 60.0
        
        decay_amount = self.decay_rate * elapsed_minutes
        
        self.valence = self._decay_value(self.valence, decay_amount)
        self.arousal = self._decay_value(self.arousal, decay_amount)
        self.dominance = self._decay_value(self.dominance, decay_amount)
        
        self.last_update = now

    def _decay_value(self, value, amount):
        if value > 0:
            return max(0, value - amount)
        elif value < 0:
            return min(0, value + amount)
        return 0

    def get_mood_label(self):
        """Map VAD to the 8 categories."""
        # Simple mapping for now
        if self.arousal > 0.5 and self.valence < -0.3: return "stressed"
        if self.valence < -0.5 and self.dominance > 0.3: return "angry"
        if self.arousal > 0.5 and self.valence > 0.5: return "excited"
        if self.valence > 0.4: return "happy"
        if self.arousal < -0.4 and self.valence > 0.2: return "calm"
        if self.arousal < -0.5 and self.valence < -0.2: return "tired"
        if self.valence < -0.4: return "sad"
        return "neutral"

    def save_state(self):
        os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
        with open(self.STATE_FILE, 'w') as f:
            json.dump({
                "valence": self.valence,
                "arousal": self.arousal,
                "dominance": self.dominance,
                "last_update": self.last_update
            }, f)

    def load_state(self):
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.valence = data.get("valence", 0.0)
                    self.arousal = data.get("arousal", 0.0)
                    self.dominance = data.get("dominance", 0.0)
                    self.last_update = data.get("last_update", time.time())
                self.apply_decay()
            except Exception:
                pass

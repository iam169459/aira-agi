"""Core Agent Brain integrating mood, self-healing, safe-mode, and actions."""
from purple_ai.mood_engine.mood_state import MoodState
from purple_ai.self_healing.self_heal import SelfHealingEngine
from purple_ai.safe_mode.guardrails import SafeModeGuard
from purple_ai.action.voice_io import VoiceEngine
from purple_ai.perception.emotion_perception import EmotionPerception


class PurpleAgent:
    """Self-healing, emotion-aware AI agent with safe-mode guardrails."""

    def __init__(self, mood_path="mood.json", safe=True):
        self.mood = MoodState.load(mood_path) if mood_path else MoodState()
        self.guard = SafeModeGuard(safe=safe)
        self.heal = SelfHealingEngine()
        self.voice = VoiceEngine()
        self.voice.mood = self.mood
        self.perception = EmotionPerception()
        self.mood_path = mood_path or "purple_mood.json"

    def observe_audio(self, audio_path: str) -> None:
        """Parse audio emotion and update persistent mood."""
        result = self.perception.detect(audio_path)
        new_mood_dict = {
            "valence": result.get("valence", 0.0),
            "arousal": result.get("arousal", 0.5),
            "dominance": result.get("dominance", 0.5),
        }
        emotion = self.mood.update(new_mood_dict)
        self.mood.save(self.mood_path)
        print(f"[Mood updated] {self.mood.describe()}")

    def respond(self, prompt: str) -> str:
        """Generate adaptive response based on current mood."""
        emotion = self.mood.emotion
        # Adaptive response styling based on current mood
        tone_words = {
            "calm": "calmly",
            "happy": "cheerfully",
            "excited": "with enthusiasm",
            "stressed": "carefully",
            "angry": "firmly but respectfully",
            "tired": "gently",
            "sad": "softly",
        }
        tone = tone_words.get(emotion, "thoughtfully")
        # In future integration: connect to core model for actual text generation
        response_text = f"(Response in a {tone} tone, emotion: {emotion}) {prompt}"
        return response_text

    def self_check(self, source_path: str) -> list:
        with open(source_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        errors = self.heal.diagnose_code(source)
        if errors:
            print(f"Self-healing: found {len(errors)} issue(s) in {source_path}")
            for err in errors:
                print(f"  - {err['msg']}")
        else:
            print("Self-healing: no issues detected.")
        return errors

    def safe_execute(self, action_description: str, fun) -> bool:
        if not self.guard.check_text(str(action_description)):
            print(f"[SAFE MODE BLOCKED] Action: {action_description}")
            return False
        try:
            fun()
            return True
        except Exception as e:
            print(f"[SAFE MODE ERROR] {e}")
            return False

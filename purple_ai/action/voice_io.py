"""Voice-to-voice conversation module using Whisper + pyttsx3."""
import tempfile, os
from typing import Optional


class VoiceEngine:
    """Voice-to-voice loop with emotion-aware TTS responses."""

    def __init__(self, stt_size="base", tts_rate=150):
        self.mood = None  # injected by agent
        self.stt = self._load_stt(stt_size)
        self.tts = self._load_tts(tts_rate)

    def _load_stt(self, size):
        try:
            from modules.stt import STTModule
            return STTModule(model_size=size)
        except Exception as e:
            print(f"STT unavailable: {e}")
            return None

    def _load_tts(self, rate):
        try:
            from modules.tts import TTSModule
            return TTSModule(rate=rate, use_offline=True)
        except Exception as e:
            print(f"TTS unavailable: {e}")
            return None

    def listen_and_respond(self, audio_input_path: str, safe_guard) -> Optional[str]:
        if not self.stt:
            return None
        result = self.stt.transcribe(audio_input_path)
        text_input = result.get("text", "")
        # Adaptive responses: adapt tone based on mood
        emotion = self.mood.emotion if self.mood else "neutral"
        # Response generation is handled by the core agent; here we just prepare TTS
        return text_input, emotion

    def speak_text(self, text: str, emotion: str = "neutral") -> Optional[str]:
        if self.tts:
            return self.tts.speak(text, emotion=emotion)
        print(f"[TTS output - {emotion}]: {text}")
        return None

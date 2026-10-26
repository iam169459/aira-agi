"""
Text-to-Speech Module
Uses pyttsx3 for offline TTS with emotion adaptation
"""
import pyttsx3
import numpy as np
import tempfile
import os
from typing import Optional


class TTSModule:
    """Text-to-Speech using pyttsx3"""
    
    def __init__(self, rate=150, volume=0.9, voice_id=None, use_offline=False, young_female=True):
        """
        Initialize TTS engine
        rate: words per minute
        volume: 0.0 to 1.0
        voice_id: None for default, or specific voice index
        use_offline: whether to try pyttsx3 (may fail in headless environments)
        young_female: try to use a young female voice
        """
        self.use_offline = use_offline
        self.rate = rate
        self.volume = volume
        self.young_female = young_female
        
        # Try to initialize pyttsx3, but handle failures gracefully
        self.engine = None
        self._init_successful = False
        
        if use_offline:
            try:
                self.engine = pyttsx3.init()
                # Young girl-like voice settings (17-19 years)
                self.engine.setProperty('rate', max(165, rate))
                self.engine.setProperty('volume', volume)
                
                # Try to select a female/young voice
                voices = self.engine.getProperty('voices')
                selected = False
                if voices:
                    # Look for female voices
                    for i, voice in enumerate(voices):
                        name = str(getattr(voice, 'name', '')).lower()
                        if any(x in name for x in ['female', 'zira', 'hazel', 'samantha', 'victoria', 'karen']):
                            self.engine.setProperty('voice', voice.id)
                            selected = True
                            break
                    if not selected:
                        # Try to pick a different voice (often second is female on some systems)
                        if len(voices) > 1:
                            self.engine.setProperty('voice', voices[1].id)
                
                self._init_successful = True
                print("pyttsx3 TTS engine initialized successfully (young female voice preference)")
            except Exception as e:
                print(f"pyttsx3 initialization failed: {e}")
                print("TTS will use text output only")
                self.engine = None
                self._init_successful = False
        else:
            print("TTS mode: text output only (offline disabled)")
            self._init_successful = False
    
    def speak(self, text: str, emotion: str = "neutral") -> Optional[str]:
        """
        Speak text with optional emotion adaptation
        Returns path to saved audio file, or None if TTS not available
        """
        if self.engine is None or not self._init_successful:
            # Print text instead of speaking
            print(f"[Aira]: {text} (emotion: {emotion})")
            return None
        
        try:
            # Modify voice properties based on emotion - young female voice
            emotion_modifiers = {
                "happy": {"rate": 175, "volume": 1.0},
                "sad": {"rate": 145, "volume": 0.75},
                "angry": {"rate": 185, "volume": 0.95},
                "excited": {"rate": 195, "volume": 1.0},
                "calm": {"rate": 155, "volume": 0.85},
                "neutral": {"rate": 170, "volume": 0.9},
                "flirty": {"rate": 165, "volume": 0.9},
                "cheerful": {"rate": 180, "volume": 1.0}
            }
            
            modifier = emotion_modifiers.get(emotion, emotion_modifiers["neutral"])
            self.engine.setProperty('rate', modifier["rate"])
            self.engine.setProperty('volume', modifier["volume"])
            
            # Save to temp file
            if not hasattr(self, 'temp_dir'):
                import tempfile
                self.temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(self.temp_dir, f"speech_{hash(text) % 10000}.wav")
            self.engine.save_to_file(text, audio_path)
            self.engine.runAndWait()
            
            return audio_path
            
        except Exception as e:
            print(f"TTS error: {e}")
            # Fall back to printing
            print(f"[Aira]: {text}")
            return None
    
    def speak_immediate(self, text: str, emotion: str = "neutral"):
        """Speak text immediately (blocking)"""
        self.speak(text, emotion)
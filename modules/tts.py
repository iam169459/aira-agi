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
    
    def __init__(self, rate=150, volume=0.9, voice_id=None, use_offline=False):
        """
        Initialize TTS engine
        rate: words per minute
        volume: 0.0 to 1.0
        voice_id: None for default, or specific voice index
        use_offline: whether to try pyttsx3 (may fail in headless environments)
        """
        self.use_offline = use_offline
        self.rate = rate
        self.volume = volume
        
        # Try to initialize pyttsx3, but handle failures gracefully
        self.engine = None
        self._init_successful = False
        
        if use_offline:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', rate)
                self.engine.setProperty('volume', volume)
                
                # Set voice
                voices = self.engine.getProperty('voices')
                if voice_id is not None and voice_id < len(voices):
                    self.engine.setProperty('voice', voices[voice_id].id)
                
                self._init_successful = True
                print("pyttsx3 TTS engine initialized successfully")
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
            print(f"[TTS would speak]: {text} (emotion: {emotion})")
            return None
        
        try:
            # Modify voice properties based on emotion
            emotion_modifiers = {
                "happy": {"rate": 160, "volume": 1.0},
                "sad": {"rate": 120, "volume": 0.7},
                "angry": {"rate": 180, "volume": 0.9},
                "excited": {"rate": 200, "volume": 1.0},
                "calm": {"rate": 130, "volume": 0.8},
                "neutral": {"rate": 150, "volume": 0.9}
            }
            
            modifier = emotion_modifiers.get(emotion, emotion_modifiers["neutral"])
            self.engine.setProperty('rate', modifier["rate"])
            self.engine.setProperty('volume', modifier["volume"])
            
            # Save to temp file
            audio_path = os.path.join(self.temp_dir, f"speech_{hash(text) % 10000}.wav")
            self.engine.save_to_file(text, audio_path)
            self.engine.runAndWait()
            
            return audio_path
            
        except Exception as e:
            print(f"TTS error: {e}")
            # Fall back to printing
            print(f"[TTS fallback]: {text}")
            return None
    
    def speak_immediate(self, text: str, emotion: str = "neutral"):
        """Speak text immediately (blocking)"""
        self.speak(text, emotion)
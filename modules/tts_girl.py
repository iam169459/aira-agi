"""
Enhanced Text-to-Speech Module for Girl Voice

This module provides a more girl-like, natural, and expressive voice for Aira.
It enhances pyttsx3 with:
- Higher pitch voice selection
- More expressive speech patterns
- Emotion-based voice modulation
- Girl-specific voice characteristics
"""

import pyttsx3
import numpy as np
import tempfile
import os
import sys
import time
from typing import Optional, Dict, Any
import platform


class GirlVoiceTTS:
    """
    Enhanced TTS with girl-like voice characteristics
    """
    
    def __init__(self, 
                 rate: int = 175, 
                 volume: float = 0.9, 
                 pitch: int = 110,
                 use_offline: bool = True,
                 ultra_girl: bool = True):
        """
        Initialize Girl Voice TTS
        
        Args:
            rate: Words per minute (higher = faster)
            volume: Volume level (0.0 to 1.0)
            pitch: Voice pitch (higher = more feminine)
            use_offline: Use pyttsx3 offline TTS
            ultra_girl: Use ultra-girl voice settings
        """
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.use_offline = use_offline
        self.ultra_girl = ultra_girl
        
        self.engine = None
        self._init_successful = False
        self.temp_dir = None
        self.current_emotion = "neutral"
        self.voice_id = None
        
        # Platform-specific settings
        self.platform = platform.system().lower()
        
        # Initialize engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize TTS engine with girl voice"""
        try:
            self.engine = pyttsx3.init()
            self._init_successful = True
            
            # Set base properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Try to set pitch (if supported)
            try:
                self.engine.setProperty('pitch', self.pitch)
            except:
                pass  # Not all engines support pitch
            
            # Find and select the best girl voice
            self._select_girl_voice()
            
            # Apply ultra-girl settings if enabled
            if self.ultra_girl:
                self._apply_ultra_girl_settings()
            
            print(f"✅ Girl Voice TTS initialized successfully!")
            print(f"   Voice: {self._get_current_voice_name()}")
            print(f"   Rate: {self.rate} WPM, Volume: {self.volume}, Pitch: {self.pitch}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Girl Voice TTS: {e}")
            self._init_successful = False
    
    def _select_girl_voice(self):
        """Select the most girl-like voice available"""
        try:
            voices = self.engine.getProperty('voices')
            
            if not voices:
                print("⚠️ No voices available")
                return
            
            # Voice selection strategy based on platform
            if self.platform == 'windows':
                # Windows: Look for specific female voices
                voice_keywords = [
                    'zira',  # Microsoft Zira (female)
                    'hazel',  # Microsoft Hazel (female, UK)
                    'samantha',  # macOS Samantha
                    'victoria',  # Microsoft Victoria (female, US)
                    'karen',  # Australian female
                    'female',
                    'woman',
                ]
                
                for voice in voices:
                    voice_name = str(voice.name).lower()
                    if any(keyword in voice_name for keyword in voice_keywords):
                        self.voice_id = voice.id
                        self.engine.setProperty('voice', voice.id)
                        print(f"   Selected voice: {voice.name}")
                        return
                
                # Fallback: Try second voice (often female)
                if len(voices) > 1:
                    self.voice_id = voices[1].id
                    self.engine.setProperty('voice', voices[1].id)
                    print(f"   Fallback voice: {voices[1].name}")
                    return
            
            elif self.platform == 'darwin':  # macOS
                # macOS: Look for female voices
                voice_keywords = ['samantha', 'alex', 'victoria', 'female']
                
                for voice in voices:
                    voice_name = str(voice.name).lower()
                    if any(keyword in voice_name for keyword in voice_keywords):
                        self.voice_id = voice.id
                        self.engine.setProperty('voice', voice.id)
                        print(f"   Selected voice: {voice.name}")
                        return
            
            else:  # Linux and others
                # Try to find any female voice
                voice_keywords = ['female', 'woman', 'girl']
                
                for voice in voices:
                    voice_name = str(voice.name).lower()
                    voice_id_str = str(voice.id).lower()
                    if any(keyword in voice_name or keyword in voice_id_str for keyword in voice_keywords):
                        self.voice_id = voice.id
                        self.engine.setProperty('voice', voice.id)
                        print(f"   Selected voice: {voice.name}")
                        return
                
                # Fallback: Try second voice
                if len(voices) > 1:
                    self.voice_id = voices[1].id
                    self.engine.setProperty('voice', voices[1].id)
                    print(f"   Fallback voice: {voices[1].name}")
                    return
            
            # If no female voice found, use default
            if voices:
                self.voice_id = voices[0].id
                self.engine.setProperty('voice', voices[0].id)
                print(f"   Default voice: {voices[0].name}")
                
        except Exception as e:
            print(f"⚠️ Could not select girl voice: {e}")
    
    def _apply_ultra_girl_settings(self):
        """Apply ultra-girl voice settings for maximum femininity"""
        try:
            # Windows-specific ultra-girl settings
            if self.platform == 'windows':
                # Try to use SAPI5 with specific voice
                try:
                    import winreg
                    # Check if we can access Windows registry for voice settings
                except:
                    pass
            
            # General ultra-girl settings
            # Increase pitch slightly more
            try:
                self.engine.setProperty('pitch', min(self.pitch + 10, 200))
            except:
                pass
            
            # Slightly faster rate for more energetic feel
            self.engine.setProperty('rate', min(self.rate + 10, 250))
            
            print("   ✨ Ultra-girl voice settings applied!")
            
        except Exception as e:
            print(f"⚠️ Could not apply ultra-girl settings: {e}")
    
    def _get_current_voice_name(self) -> str:
        """Get the name of the current voice"""
        try:
            voices = self.engine.getProperty('voices')
            current_voice_id = self.engine.getProperty('voice')
            for voice in voices:
                if voice.id == current_voice_id:
                    return str(voice.name)
            return "Unknown"
        except:
            return "Unknown"
    
    def set_emotion(self, emotion: str):
        """
        Set the current emotion for voice modulation
        
        Args:
            emotion: Emotion name (happy, sad, angry, excited, calm, etc.)
        """
        self.current_emotion = emotion.lower()
        
        # Apply emotion-specific voice modulation
        emotion_settings = {
            "happy": {"rate": self.rate + 20, "volume": min(self.volume + 0.1, 1.0), "pitch": self.pitch + 10},
            "excited": {"rate": self.rate + 30, "volume": min(self.volume + 0.15, 1.0), "pitch": self.pitch + 15},
            "sad": {"rate": self.rate - 30, "volume": max(self.volume - 0.2, 0.3), "pitch": self.pitch - 15},
            "angry": {"rate": self.rate + 25, "volume": min(self.volume + 0.1, 1.0), "pitch": self.pitch - 5},
            "calm": {"rate": self.rate - 20, "volume": max(self.volume - 0.1, 0.4), "pitch": self.pitch - 10},
            "shy": {"rate": self.rate - 25, "volume": max(self.volume - 0.25, 0.3), "pitch": self.pitch - 10},
            "playful": {"rate": self.rate + 25, "volume": min(self.volume + 0.05, 1.0), "pitch": self.pitch + 10},
            "curious": {"rate": self.rate + 10, "volume": self.volume, "pitch": self.pitch + 5},
            "surprised": {"rate": self.rate + 35, "volume": min(self.volume + 0.2, 1.0), "pitch": self.pitch + 20},
            "flirty": {"rate": self.rate - 10, "volume": max(self.volume - 0.05, 0.5), "pitch": self.pitch + 5},
            "embarrassed": {"rate": self.rate - 15, "volume": max(self.volume - 0.2, 0.4), "pitch": self.pitch - 5},
            "proud": {"rate": self.rate + 15, "volume": min(self.volume + 0.1, 1.0), "pitch": self.pitch + 10},
            "disappointed": {"rate": self.rate - 20, "volume": max(self.volume - 0.15, 0.4), "pitch": self.pitch - 10},
            "neutral": {"rate": self.rate, "volume": self.volume, "pitch": self.pitch},
        }
        
        settings = emotion_settings.get(emotion, emotion_settings["neutral"])
        
        try:
            self.engine.setProperty('rate', settings["rate"])
            self.engine.setProperty('volume', settings["volume"])
            try:
                self.engine.setProperty('pitch', settings["pitch"])
            except:
                pass
        except Exception as e:
            print(f"⚠️ Could not apply emotion settings: {e}")
    
    def speak(self, text: str, emotion: str = None, block: bool = True) -> Optional[str]:
        """
        Speak text with girl voice
        
        Args:
            text: Text to speak
            emotion: Emotion for voice modulation (optional)
            block: Whether to block until speech completes
            
        Returns:
            Path to saved audio file, or None if TTS not available
        """
        if not self._init_successful or self.engine is None:
            # Fallback to text output
            print(f"[Aira (girl voice)]: {text}")
            return None
        
        # Set emotion if provided
        if emotion:
            self.set_emotion(emotion)
        
        try:
            # Create temp directory if needed
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()
            
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            audio_path = os.path.join(self.temp_dir, f"aira_{timestamp}.wav")
            
            # Save to file
            self.engine.save_to_file(text, audio_path)
            
            # Speak
            if block:
                self.engine.runAndWait()
            else:
                self.engine.startLoop(False)
                self.engine.iterate()
            
            return audio_path
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            # Fallback to text output
            print(f"[Aira (girl voice)]: {text}")
            return None
    
    def speak_immediate(self, text: str, emotion: str = None):
        """
        Speak text immediately (blocking)
        
        Args:
            text: Text to speak
            emotion: Emotion for voice modulation
        """
        self.speak(text, emotion, block=True)
    
    def stop(self):
        """Stop current speech"""
        try:
            if self.engine:
                self.engine.stop()
        except:
            pass
    
    def is_available(self) -> bool:
        """Check if TTS is available"""
        return self._init_successful and self.engine is not None
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about the current voice"""
        try:
            voices = self.engine.getProperty('voices')
            current_voice_id = self.engine.getProperty('voice')
            
            voice_info = {
                "current_voice": self._get_current_voice_name(),
                "voice_id": current_voice_id,
                "rate": self.engine.getProperty('rate'),
                "volume": self.engine.getProperty('volume'),
                "available_voices": [str(v.name) for v in voices],
                "platform": self.platform,
                "ultra_girl": self.ultra_girl,
            }
            
            try:
                voice_info["pitch"] = self.engine.getProperty('pitch')
            except:
                voice_info["pitch"] = "Not supported"
            
            return voice_info
        except:
            return {"error": "Could not get voice info"}
    
    def list_available_voices(self) -> list:
        """List all available voices"""
        try:
            voices = self.engine.getProperty('voices')
            return [{
                "id": voice.id,
                "name": str(voice.name),
                "languages": str(voice.languages),
                "gender": self._detect_gender(str(voice.name))
            } for voice in voices]
        except:
            return []
    
    def _detect_gender(self, voice_name: str) -> str:
        """Detect gender from voice name"""
        voice_name_lower = voice_name.lower()
        
        if any(kw in voice_name_lower for kw in ['female', 'woman', 'girl', 'lady', 'zira', 'hazel', 'samantha', 'victoria', 'karen']):
            return "female"
        elif any(kw in voice_name_lower for kw in ['male', 'man', 'boy', 'gentleman', 'david', 'mark', 'daniel']):
            return "male"
        else:
            return "unknown"
    
    def set_voice_by_index(self, index: int):
        """Set voice by index"""
        try:
            voices = self.engine.getProperty('voices')
            if 0 <= index < len(voices):
                self.voice_id = voices[index].id
                self.engine.setProperty('voice', voices[index].id)
                print(f"Voice set to: {voices[index].name}")
                return True
            return False
        except Exception as e:
            print(f"Could not set voice: {e}")
            return False
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
        except:
            pass


# Create a global instance for easy access
girl_tts = GirlVoiceTTS()


if __name__ == "__main__":
    # Test the girl voice TTS
    print("Testing Girl Voice TTS")
    print("=" * 50)
    
    tts = GirlVoiceTTS(ultra_girl=True)
    
    if tts.is_available():
        print("\nVoice Info:")
        info = tts.get_voice_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\nAvailable Voices:")
        voices = tts.list_available_voices()
        for i, voice in enumerate(voices):
            print(f"  {i}: {voice['name']} ({voice['gender']})")
        
        print("\nTesting different emotions:")
        emotions = ["neutral", "happy", "excited", "sad", "playful", "shy"]
        for emotion in emotions:
            print(f"\n  Testing {emotion}:")
            tts.speak(f"This is me speaking with a {emotion} voice!", emotion=emotion)
            time.sleep(0.5)
    else:
        print("TTS not available. Falling back to text output.")
        print("\nText output test:")
        tts.speak("Hello! I'm Aira, your cute AI assistant!")

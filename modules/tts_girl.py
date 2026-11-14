"""
Enhanced Text-to-Speech Module for Girl Voice

This module provides a more girl-like, natural, and expressive voice for Aira.
It enhances pyttsx3 with:
- Higher pitch voice selection
- More expressive speech patterns
- Emotion-based voice modulation
- Girl-specific voice characteristics

Human-like speech layer:
- Breath groups with natural pauses and rhythm
- Synthesized breathing sounds between phrases
- Fillers, hesitations and contractions ("um", "hmm", "I'm", ...)
- Streaming playback: first phrase is spoken while the rest renders
"""

import pyttsx3
import numpy as np
import tempfile
import os
import sys
import time
import random
import re
import wave
from typing import Optional, Dict, Any, List, Tuple
import platform

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    sd = None
    SD_AVAILABLE = False


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
        
        # Human-like speech settings
        self.human_like = True
        self.breathing = True
        self.filler_chance = 0.22
        self.hesitation_chance = 0.10
        self.laugh_chance = 0.12
        self.breath_chance = 0.28
        self._rng = random.Random()
        self._nprng = np.random.default_rng()
        
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
    
    # ------------------------------------------------------------------
    # Human-like speech layer
    # ------------------------------------------------------------------
    
    _CONTRACTIONS = [
        (r"\bI am\b", "I'm"), (r"\bi am\b", "I'm"),
        (r"\byou are\b", "you're"), (r"\bYou are\b", "You're"),
        (r"\bwe are\b", "we're"), (r"\bWe are\b", "We're"),
        (r"\bthey are\b", "they're"), (r"\bThat is\b", "That's"),
        (r"\bthat is\b", "that's"), (r"\bIt is\b", "It's"),
        (r"\bit is\b", "it's"), (r"\bWhat is\b", "What's"),
        (r"\bwhat is\b", "what's"), (r"\bdo not\b", "don't"),
        (r"\bDo not\b", "Don't"), (r"\bdoes not\b", "doesn't"),
        (r"\bis not\b", "isn't"), (r"\bare not\b", "aren't"),
        (r"\bcannot\b", "can't"), (r"\bcan not\b", "can't"),
        (r"\bwill not\b", "won't"), (r"\bwould not\b", "wouldn't"),
        (r"\bI will\b", "I'll"), (r"\bI have\b", "I've"),
        (r"\bI would\b", "I'd"), (r"\bgoing to\b", "gonna"),
        (r"\bwant to\b", "wanna"), (r"\bkind of\b", "kinda"),
    ]
    
    _STARTERS = ["Hmm,", "Well,", "Okay so,", "You know,", "Mmm,", "Let's see,"]
    
    _HESITATIONS = ["um", "like", "I mean", "you know", "uhm"]
    
    _PAUSE_MOD = {
        "excited": 0.70, "surprised": 0.75, "playful": 0.80, "happy": 0.85,
        "neutral": 1.0, "flirty": 1.05, "embarrassed": 1.15, "shy": 1.20,
        "calm": 1.30, "sad": 1.45,
    }
    
    _BREATH_MOD = {
        "excited": 0.4, "surprised": 0.5, "playful": 0.6, "happy": 0.7,
        "neutral": 1.0, "flirty": 1.1, "shy": 1.2, "embarrassed": 1.2,
        "calm": 1.5, "sad": 1.6,
    }
    
    def _humanize_text(self, text: str, emotion: str = None) -> str:
        """Add contractions, fillers and hesitations so the reply sounds spoken, not read."""
        result = text.strip()
        if not result:
            return result
        
        for pattern, replacement in self._CONTRACTIONS:
            result = re.sub(pattern, replacement, result)
        
        starter_chance = self.filler_chance
        if emotion in ("shy", "embarrassed", "sad"):
            starter_chance += 0.10
        elif emotion in ("excited", "surprised"):
            starter_chance -= 0.12
        
        if self._rng.random() < starter_chance:
            starter = self._rng.choice(self._STARTERS)
            words = result.split()
            first = words[0]
            if first not in ("I", "I'm", "I'll", "I've", "I'd", "Oh", "Okay", "Yes", "No", "You"):
                result = f"{starter} {first[0].lower() + first[1:]}{' ' + ' '.join(words[1:]) if len(words) > 1 else ''}"
            else:
                result = f"{starter} {result}"
        
        if self._rng.random() < self.hesitation_chance:
            commas = [m.start() + 1 for m in re.finditer(r",", result)]
            if commas:
                pos = self._rng.choice(commas)
                filler = self._rng.choice(self._HESITATIONS)
                result = f"{result[:pos]} {filler},{result[pos:]}"
        
        if (emotion in ("happy", "excited", "playful", "flirty")
                and result.endswith(("!", "?"))
                and self._rng.random() < self.laugh_chance):
            result = f"{result} Hehe!"
        
        if result[-1] not in ".!?…,":
            result += "."
        
        return result
    
    def _split_breath_groups(self, text: str, max_words: int = 10) -> List[str]:
        """Split text into short breath-sized groups a person would say in one breath."""
        groups: List[str] = []
        
        for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
            if not sentence:
                continue
            buf = ""
            for clause in re.split(r"(?<=,)\s+", sentence):
                for piece in self._split_long_clause(clause, max_words):
                    candidate = f"{buf} {piece}".strip()
                    if buf and len(candidate.split()) > max_words:
                        groups.append(buf)
                        buf = piece
                    else:
                        buf = candidate
            if buf:
                groups.append(buf)
        
        return groups
    
    def _split_long_clause(self, clause: str, max_words: int) -> List[str]:
        """Break comma-less clauses into smaller chunks."""
        words = clause.split()
        if len(words) <= max_words * 1.6:
            return [clause]
        size = max(4, max_words // 2 + 2)
        return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    
    def _group_pause(self, group: str, emotion: str = None) -> float:
        """Natural silence after a group: longer at sentence ends, modulated by emotion."""
        base = 0.38 if group.rstrip().endswith((".", "!", "?", "…")) else 0.16
        mod = self._PAUSE_MOD.get((emotion or "neutral").lower(), 1.0)
        return base * mod * (0.85 + 0.30 * self._rng.random())
    
    def _make_breath(self, sr: int = 22050, duration: float = 0.32, strength: float = 1.0) -> np.ndarray:
        """Synthesize a soft inhale: smoothed noise with a rising envelope."""
        n = max(1, int(sr * duration))
        noise = self._nprng.standard_normal(n)
        win = max(3, int(sr * 0.004))
        noise = np.convolve(noise, np.ones(win) / win, mode="same")
        t = np.linspace(0.0, 1.0, n)
        envelope = (np.sin(np.pi * t) ** 1.5) * (0.35 + 0.65 * t)
        return (noise * envelope * 0.12 * strength).astype(np.float32)
    
    def _load_wav(self, path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load a wav file as float32 mono samples plus its sample rate."""
        try:
            with wave.open(path, "rb") as w:
                sr = w.getframerate()
                channels = w.getnchannels()
                width = w.getsampwidth()
                raw = w.readframes(w.getnframes())
            
            if width == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif width == 4:
                audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif width == 1:
                audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                return None, None
            
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            return audio, sr
        except Exception:
            return None, None
    
    def _render_group(self, text: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Render one breath group to PCM samples via the TTS engine."""
        try:
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()
            path = os.path.join(self.temp_dir, f"aira_{int(time.time() * 1000000)}.wav")
            self.engine.save_to_file(text, path)
            self.engine.runAndWait()
            audio, sr = self._load_wav(path)
            try:
                os.unlink(path)
            except OSError:
                pass
            return audio, sr
        except Exception as e:
            print(f"⚠️ Could not render speech group: {e}")
            return None, None
    
    def _save_combined(self, pieces: List[np.ndarray], sr: int) -> Optional[str]:
        """Concatenate rendered pieces (speech, breaths, silences) into one wav file."""
        try:
            pieces = [p for p in pieces if p is not None and len(p) > 0]
            if not pieces:
                return None
            combined = np.concatenate(pieces)
            pcm = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()
            path = os.path.join(self.temp_dir, f"aira_{int(time.time() * 1000000)}.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm.tobytes())
            return path
        except Exception as e:
            print(f"⚠️ Could not save combined audio: {e}")
            return None
    
    def speak(self, text: str, emotion: str = None, block: bool = True, play: bool = False) -> Optional[str]:
        """
        Speak text with a human-like rhythm.
        
        The text is split into breath groups. Between groups Aira takes short
        pauses and occasionally audibly breathes in. With play=True the first
        group is played the moment it is rendered, so she answers faster.
        
        Args:
            text: Text to speak
            emotion: Emotion for voice modulation (optional)
            block: Whether to block until speech completes
            play: Stream audio through the speakers inside this call
            
        Returns:
            Path to saved audio file, or None when streamed or unavailable
        """
        if not self._init_successful or self.engine is None:
            if not play:
                print(f"[Aira (girl voice)]: {text}")
            return None
        
        if not text or not text.strip():
            return None
        
        if emotion:
            self.set_emotion(emotion)
        
        if self.human_like:
            text = self._humanize_text(text, emotion)
        
        groups = self._split_breath_groups(text)
        if not groups:
            return None
        
        stream = play and SD_AVAILABLE
        chunks: List[np.ndarray] = []
        pieces: List[np.ndarray] = []
        sr = 22050
        breath_mod = self._BREATH_MOD.get((emotion or "neutral").lower(), 1.0)
        
        try:
            for i, group in enumerate(groups):
                audio, group_sr = self._render_group(group)
                if audio is None or group_sr is None:
                    continue
                sr = group_sr
                chunks.append(audio)
                
                if stream:
                    if i == 0 and self.breathing:
                        sd.play(self._make_breath(sr, strength=0.9), sr)
                        sd.wait()
                        time.sleep(0.05)
                    sd.play(audio, sr)
                    if block:
                        sd.wait()
                    if i < len(groups) - 1:
                        time.sleep(self._group_pause(group, emotion))
                        if self.breathing and self._rng.random() < self.breath_chance * breath_mod:
                            sd.play(self._make_breath(sr), sr)
                            if block:
                                sd.wait()
                else:
                    if i == 0 and self.breathing:
                        pieces.append(self._make_breath(sr, strength=0.9))
                        pieces.append(np.zeros(int(sr * 0.05), dtype=np.float32))
                    pieces.append(audio)
                    if i < len(groups) - 1:
                        pieces.append(np.zeros(int(sr * self._group_pause(group, emotion)), dtype=np.float32))
                        if self.breathing and self._rng.random() < self.breath_chance * breath_mod:
                            pieces.append(self._make_breath(sr))
            
            if stream:
                return None
            return self._save_combined(pieces, sr)
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            if not play:
                print(f"[Aira (girl voice)]: {text}")
            return None
    
    def speak_immediate(self, text: str, emotion: str = None):
        """
        Speak text immediately (blocking)
        
        Args:
            text: Text to speak
            emotion: Emotion for voice modulation
        """
        self.speak(text, emotion, block=True, play=True)
    
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

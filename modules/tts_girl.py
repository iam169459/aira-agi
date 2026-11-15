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


# ==================== HUMAN-LIKE SPEECH FEATURES (from local working copy) ====================

FILLERS = {
    "thinking": ["um", "uh", "let me think", "hmm", "well"],
    "uncertain": ["I guess", "sort of", "kind of", "maybe", "probably"],
    "casual": ["like", "you know", "I mean", "actually", "basically"],
    "nervous": ["uh", "um", "er", "ah"],
    "excited": ["oh!", "wow!", "amazing!", "really?"],
    "hesitant": ["um...", "uh...", "well..."],
    "agreement": ["yeah", "uh-huh", "right", "exactly", "totally"],
    "surprise": ["really?", "no way!", "wow", "seriously?"],
    "understanding": ["I see", "got it", "makes sense", "okay"],
    "neutral": ["um", "uh", "well"],
}

# Non-verbal vocalizations
NON_VERBAL = {
    "laugh": ["haha", "hehe", "hahaha", "hehehe"],
    "chuckle": ["heh", "hehe", "huh"],
    "giggle": ["hehe", "teehee", "hihi"],
    "sigh": ["*sigh*", "*deep breath*", "hhh"],
    "gasp": ["*gasp*", "oh!", "ah!"],
    "cough": ["*cough*", "ahem"],
    "sniff": ["*sniff*"],
    "yawn": ["*yawn*", "mmm"],
    "hmm": ["hmm", "hmmm"],
    "uh_huh": ["uh-huh", "mm-hmm"],
    "nuh_uh": ["nuh-uh", "mm-mm"],
}

# Emphasis patterns - words that typically get stressed
EMPHASIS_PATTERNS = [
    # Negations - always emphasized
    (r'\b(not|never|no|don\'t|can\'t|won\'t|shouldn\'t|wouldn\'t|couldn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t|didn\'t|doesn\'t)\b', 'strong'),
    # Intensifiers
    (r'\b(very|really|extremely|absolutely|totally|completely|utterly|so|such)\b', 'moderate'),
    # Question words
    (r'\b(why|how|what|when|where|who|which)\b', 'moderate'),
    # Contrastive focus markers
    (r'\b(but|however|although|though|yet|still|instead|rather)\b', 'moderate'),
    # First person emphasis
    (r'\b(I|me|my|mine)\b', 'light'),
    # Demonstratives
    (r'\b(this|that|these|those)\b', 'light'),
]

# Self-correction patterns
SELF_CORRECTIONS = [
    ("actually,", "I mean,"),
    ("wait,", "no,"),
    ("I mean,", "what I mean is,"),
    ("or rather,", "let me rephrase,"),
    ("that is,", "in other words,"),
]

# Vocal fry markers (end of phrases)
VOCAL_FRY_PROBABILITY = 0.15  # 15% chance at phrase endings


try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    sd = None
    SD_AVAILABLE = False

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except Exception:
    PiperVoice = None
    PIPER_AVAILABLE = False


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
        self._emotion_settings = {"rate": rate, "volume": volume, "pitch": pitch}
        self.piper_voice = None
        self.piper_model = None
        
        # Optional neural voice: much more realistic when a Piper model exists
        if PIPER_AVAILABLE:
            self._try_load_piper()
        
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
        self._emotion_settings = dict(settings)
        
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
    
    def _try_load_piper(self):
        """Load a Piper neural voice model if one is available."""
        try:
            candidates = []
            env_model = os.environ.get("AIRA_PIPER_MODEL")
            if env_model:
                candidates.append(env_model)
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "piper")
            if os.path.isdir(models_dir):
                candidates.extend(
                    os.path.join(models_dir, f) for f in sorted(os.listdir(models_dir)) if f.endswith(".onnx")
                )
            
            for model_path in candidates:
                if not os.path.exists(model_path):
                    continue
                config_path = model_path + ".json"
                try:
                    self.piper_voice = PiperVoice.load(model_path, config_path=config_path)
                except TypeError:
                    self.piper_voice = PiperVoice.load(model_path)
                self.piper_model = model_path
                print(f"🧠 Neural voice loaded: {os.path.basename(model_path)}")
                return
        except Exception as e:
            print(f"⚠️ Neural voice unavailable ({e}), using pyttsx3")
        self.piper_voice = None
    
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
        
        if emotion in ("excited", "playful", "happy") and self._rng.random() < 0.18:
            result = self._elongate_word(result)
        
        if result[-1] not in ".!?…,":
            result += "."
        
        return result
    
    _ELONGATIONS = {"so": "sooo", "really": "reaally", "very": "veery", "oh": "ohh", "wow": "woow"}
    
    def _elongate_word(self, text: str) -> str:
        """Stretch one word ("so" -> "sooo") the way people do when excited."""
        words = text.split()
        spots = [i for i, w in enumerate(words) if w.lower().strip(".,!?") in self._ELONGATIONS]
        if not spots:
            return text
        i = self._rng.choice(spots)
        word = words[i]
        stem = word.rstrip(".,!?")
        stretched = self._ELONGATIONS[stem.lower()]
        if stem[0].isupper():
            stretched = stretched.capitalize()
        suffix = word[len(stem):] if stem != word else ""
        words[i] = stretched + suffix
        return " ".join(words)
    
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
    
    _SOFT_STARTS = ("hmm", "well", "mmm", "okay so", "you know", "let's see", "um", "uhm", "i mean")
    
    def _apply_prosody(self, text: str):
        """Vary rate/pitch/volume around the emotion baseline so phrases don't sound identical.
        Filler phrases ("Hmm,", "well...") are spoken softer and slower, like real mumbles."""
        try:
            base = self._emotion_settings
            soft = text.strip().lower().startswith(self._SOFT_STARTS)
            rate_mult = 0.90 if soft else 0.96 + 0.08 * self._rng.random()
            vol_mult = 0.78 if soft else 0.95 + 0.07 * self._rng.random()
            pitch_mult = 0.96 + 0.08 * self._rng.random()
            
            self.engine.setProperty('rate', int(base["rate"] * rate_mult))
            self.engine.setProperty('volume', min(base["volume"] * vol_mult, 1.0))
            try:
                self.engine.setProperty('pitch', int(base["pitch"] * pitch_mult))
            except Exception:
                pass
        except Exception:
            pass
    
    def _apply_fades(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fade in/out to remove clicks between phrases and soften edges."""
        fade_in = max(1, int(sr * 0.012))
        fade_out = max(1, int(sr * 0.045))
        if len(audio) <= fade_in + fade_out:
            return audio
        audio = audio.copy()
        audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
        audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
        return audio
    
    def _render_group(self, text: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Render one breath group to PCM samples via the TTS engine."""
        try:
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()
            path = os.path.join(self.temp_dir, f"aira_{int(time.time() * 1000000)}.wav")
            
            if self.piper_voice is not None:
                with wave.open(path, "wb") as w:
                    if hasattr(self.piper_voice, "synthesize_wav"):
                        self.piper_voice.synthesize_wav(text, w)
                    else:
                        self.piper_voice.synthesize(text, w)
            else:
                self._apply_prosody(text)
                self.engine.save_to_file(text, path)
                self.engine.runAndWait()
            
            audio, sr = self._load_wav(path)
            try:
                os.unlink(path)
            except OSError:
                pass
            if audio is not None:
                audio = self._apply_fades(audio, sr or 22050)
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
    
    def speak(self, text: str, emotion: str = None, block: bool = True, play: bool = False, use_rhythm: bool = True, use_breathing: bool = True) -> Optional[str]:
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
        
        if self.human_like and use_rhythm:
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
                    if i == 0 and self.breathing and use_breathing:
                        sd.play(self._make_breath(sr, strength=0.9), sr)
                        sd.wait()
                        time.sleep(0.05)
                    sd.play(audio, sr)
                    if block:
                        sd.wait()
                    if i < len(groups) - 1:
                        time.sleep(self._group_pause(group, emotion))
                        if self.breathing and use_breathing and self._rng.random() < self.breath_chance * breath_mod:
                            sd.play(self._make_breath(sr), sr)
                            if block:
                                sd.wait()
                else:
                    if i == 0 and self.breathing and use_breathing:
                        pieces.append(self._make_breath(sr, strength=0.9))
                        pieces.append(np.zeros(int(sr * 0.05), dtype=np.float32))
                    pieces.append(audio)
                    if i < len(groups) - 1:
                        pieces.append(np.zeros(int(sr * self._group_pause(group, emotion)), dtype=np.float32))
                        if self.breathing and use_breathing and self._rng.random() < self.breath_chance * breath_mod:
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
    
    def _add_fillers(self, text: str, emotion: str = "neutral") -> str:
        """
        Add natural fillers and hesitations to text.
        Inserts at phrase boundaries with probability based on emotion.
        """
        filler_prob = {
            "thinking": 0.3,
            "uncertain": 0.25,
            "casual": 0.2,
            "nervous": 0.35,
            "excited": 0.15,
            "hesitant": 0.4,
            "neutral": 0.15,
        }
        
        prob = filler_prob.get(emotion, 0.15)
        fillers = FILLERS.get(emotion, FILLERS["neutral"])
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        result = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Add filler at start of sentence (except first)
            if i > 0 and random.random() < prob:
                filler = random.choice(fillers)
                # Capitalize if start of sentence
                if sentence[0].isupper():
                    filler = filler.capitalize()
                sentence = f"{filler}, {sentence}"
            
            # Add filler mid-sentence at commas
            if ',' in sentence and random.random() < prob * 0.5:
                parts = sentence.split(',', 1)
                filler = random.choice(fillers).lower()
                sentence = f"{parts[0]}, {filler}, {parts[1]}"
            
            result.append(sentence)
        
        return ' '.join(result)
    
    def _add_non_verbal(self, text: str, emotion: str = "neutral") -> str:
        """
        Add non-verbal vocalizations (laughter, sighs, gasps, etc.)
        """
        non_verbal_prob = {
            "happy": 0.2,
            "excited": 0.25,
            "amused": 0.3,
            "sad": 0.15,
            "tired": 0.2,
            "surprised": 0.2,
            "nervous": 0.1,
            "neutral": 0.05,
        }
        
        prob = non_verbal_prob.get(emotion, 0.05)
        
        if random.random() < prob:
            # Choose appropriate non-verbal
            if emotion in ["happy", "excited", "amused"]:
                nv = random.choice(NON_VERBAL["laugh"] + NON_VERBAL["giggle"])
            elif emotion in ["sad", "disappointed"]:
                nv = random.choice(NON_VERBAL["sigh"])
            elif emotion in ["surprised", "shocked"]:
                nv = random.choice(NON_VERBAL["gasp"])
            elif emotion in ["tired", "bored"]:
                nv = random.choice(NON_VERBAL["yawn"])
            else:
                nv = random.choice(NON_VERBAL["hmm"])
            
            # Insert at beginning or end
            if random.random() < 0.5:
                text = f"{nv} {text}"
            else:
                text = f"{text} {nv}"
        
        return text
    
    def _add_emphasis(self, text: str) -> str:
        """
        Add emphasis markers for important words.
        Returns text with special markers that will be processed during speech.
        """
        # We'll use special markers that the speech engine can interpret
        # For pyttsx3, we'll handle this by adjusting rate/pitch/volume per word
        # This method marks words for later processing
        return text
    
    def _apply_pre_boundary_lengthening(self, segment_text: str, is_phrase_end: bool, 
                                         is_sentence_end: bool) -> Dict:
        """
        Apply pre-boundary lengthening - slow down before phrase/sentence boundaries.
        Returns properties for the segment.
        """
        props = {
            'rate_modifier': 1.0,
            'pitch_modifier': 0,
            'volume_modifier': 1.0,
            'lengthen_final': False,
            'vocal_fry': False,
        }
        
        if is_sentence_end:
            # Significant lengthening at sentence end
            props['rate_modifier'] = 0.85
            props['lengthen_final'] = True
            # Vocal fry at end of declarative sentences
            if random.random() < VOCAL_FRY_PROBABILITY:
                props['vocal_fry'] = True
        elif is_phrase_end:
            # Moderate lengthening at phrase end (comma)
            props['rate_modifier'] = 0.92
            props['lengthen_final'] = True
        
        return props
    
    def _add_self_correction(self, text: str, emotion: str = "neutral") -> str:
        """
        Add natural self-corrections/repairs.
        """
        correction_prob = {
            "thinking": 0.15,
            "uncertain": 0.1,
            "casual": 0.08,
            "nervous": 0.12,
            "excited": 0.05,
            "neutral": 0.05,
        }
        
        prob = correction_prob.get(emotion, 0.05)
        
        if random.random() < prob:
            correction = random.choice(SELF_CORRECTIONS)
            # Insert at a natural break point
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            if len(sentences) > 1:
                insert_at = random.randint(0, len(sentences) - 2)
                sentences.insert(insert_at + 1, f"{correction[0]} {correction[1]}")
                text = ' '.join(sentences)
        
        return text
    
    def _process_emphasis_words(self, text: str) -> List[Tuple[str, Dict]]:
        """
        Process text and mark emphasis words with special properties.
        Returns list of (word, properties) tuples.
        """
        words = text.split()
        result = []
        
        for word in words:
            props = {
                'emphasis': 'none',
                'rate_modifier': 1.0,
                'pitch_modifier': 0,
                'volume_modifier': 1.0,
            }
            
            # Check emphasis patterns
            clean_word = word.lower().strip('.,!?;:')
            for pattern, level in EMPHASIS_PATTERNS:
                if re.search(pattern, clean_word, re.IGNORECASE):
                    if level == 'strong':
                        props['emphasis'] = 'strong'
                        props['rate_modifier'] = 0.9
                        props['pitch_modifier'] = 5
                        props['volume_modifier'] = 1.1
                    elif level == 'moderate':
                        props['emphasis'] = 'moderate'
                        props['rate_modifier'] = 0.95
                        props['pitch_modifier'] = 3
                        props['volume_modifier'] = 1.05
                    elif level == 'light':
                        props['emphasis'] = 'light'
                        props['rate_modifier'] = 0.98
                        props['pitch_modifier'] = 2
                        props['volume_modifier'] = 1.02
                    break
            
            result.append((word, props))
        
        return result
    
    # ==================== RHYTHM & BREATHING SYSTEM ====================
    
    def _preprocess_text_for_rhythm(self, text: str) -> List[Tuple[str, Dict]]:
        """
        Preprocess text to add natural rhythm markers.
        Returns list of (text_segment, properties) tuples.
        
        Properties include:
        - pause_after: pause duration in seconds
        - rate_modifier: speech rate multiplier
        - pitch_modifier: pitch adjustment
        - volume_modifier: volume adjustment
        - breath: whether to add breath sound
        """
        segments = []
        
        # Split into sentences (preserving punctuation)
        # This regex keeps the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # Determine if this is the last sentence
            is_last = (i == len(sentences) - 1)
            
            # Base properties for this sentence
            props = {
                'pause_after': 0.0,
                'rate_modifier': 1.0,
                'pitch_modifier': 0,
                'volume_modifier': 1.0,
                'breath': False,
            }
            
            # Add sentence-ending pause
            if sentence.endswith('.'):
                props['pause_after'] = 0.6 + random.uniform(0.1, 0.3)  # Longer pause after period
                props['breath'] = random.random() < 0.3  # 30% chance of breath
            elif sentence.endswith('?'):
                props['pause_after'] = 0.5 + random.uniform(0.1, 0.2)
                props['pitch_modifier'] = 5  # Rising intonation
                props['breath'] = random.random() < 0.2
            elif sentence.endswith('!'):
                props['pause_after'] = 0.4 + random.uniform(0.1, 0.2)
                props['rate_modifier'] = 1.05  # Slightly faster for excitement
                props['breath'] = random.random() < 0.25
            else:
                props['pause_after'] = 0.3 + random.uniform(0.05, 0.15)
            
            # Add comma pauses within sentence
            sub_segments = self._split_by_commas(sentence)
            
            for j, sub_seg in enumerate(sub_segments):
                sub_props = props.copy()
                sub_props['pause_after'] = 0.0
                
                if j < len(sub_segments) - 1:
                    # Comma pause
                    sub_props['pause_after'] = 0.25 + random.uniform(0.05, 0.1)
                    sub_props['breath'] = random.random() < 0.1  # 10% chance at commas
                
                # Add natural rhythm variation within sentence
                # Slightly slow down at the end of phrases
                if j == len(sub_segments) - 1 and not is_last:
                    sub_props['rate_modifier'] *= 0.95
                
                segments.append((sub_seg.strip(), sub_props))
        
        return segments
    
    def _split_by_commas(self, text: str) -> List[str]:
        """Split text by commas, preserving the commas"""
        # Split by commas but keep them
        parts = re.split(r'(,[^\s])', text)
        result = []
        current = ""
        
        for part in parts:
            if part.startswith(','):
                current += part
                result.append(current)
                current = ""
            else:
                current += part
        
        if current:
            result.append(current)
        
        # If no commas, return as single segment
        if len(result) == 1 and ',' not in text:
            return [text]
        
        return [s for s in result if s.strip()]
    
    def _generate_breath_pattern(self, emotion: str = "neutral") -> Dict:
        """
        Generate breathing pattern based on emotion.
        Returns dict with breath properties.
        """
        breath_patterns = {
            "neutral": {"duration": 0.4, "volume": 0.3, "pitch": 0},
            "happy": {"duration": 0.3, "volume": 0.35, "pitch": 10},
            "excited": {"duration": 0.2, "volume": 0.4, "pitch": 15},
            "sad": {"duration": 0.6, "volume": 0.25, "pitch": -10},
            "angry": {"duration": 0.25, "volume": 0.35, "pitch": -5},
            "calm": {"duration": 0.7, "volume": 0.2, "pitch": -5},
            "shy": {"duration": 0.5, "volume": 0.2, "pitch": -5},
            "playful": {"duration": 0.35, "volume": 0.3, "pitch": 5},
            "curious": {"duration": 0.4, "volume": 0.3, "pitch": 5},
            "surprised": {"duration": 0.25, "volume": 0.45, "pitch": 20},
            "flirty": {"duration": 0.5, "volume": 0.25, "pitch": 5},
            "embarrassed": {"duration": 0.45, "volume": 0.2, "pitch": -5},
            "proud": {"duration": 0.4, "volume": 0.3, "pitch": 10},
            "disappointed": {"duration": 0.55, "volume": 0.2, "pitch": -10},
        }
        
        base = breath_patterns.get(emotion, breath_patterns["neutral"])
        
        # Add natural variation
        return {
            "duration": base["duration"] + random.uniform(-0.1, 0.1),
            "volume": max(0.1, min(0.5, base["volume"] + random.uniform(-0.05, 0.05))),
            "pitch": base["pitch"] + random.randint(-3, 3),
        }
    
    def _apply_rhythm_variation(self, base_rate: int, segment_props: Dict, 
                                 emotion: str, segment_index: int, total_segments: int) -> int:
        """
        Apply natural rhythm variation to speech rate.
        """
        rate = base_rate
        
        # Apply segment-specific modifier
        rate = int(rate * segment_props.get('rate_modifier', 1.0))
        
        # Add natural variation (±5%)
        rate = int(rate * random.uniform(0.95, 1.05))
        
        # Emotion-based global modifier
        emotion_rate_mod = {
            "happy": 1.05,
            "excited": 1.15,
            "sad": 0.85,
            "angry": 1.1,
            "calm": 0.9,
            "shy": 0.88,
            "playful": 1.08,
            "curious": 1.02,
            "surprised": 1.12,
            "flirty": 0.95,
            "embarrassed": 0.9,
            "proud": 1.05,
            "disappointed": 0.88,
            "neutral": 1.0,
        }
        rate = int(rate * emotion_rate_mod.get(emotion, 1.0))
        
        # Beginning of speech - slightly slower (warm up)
        if segment_index == 0:
            rate = int(rate * 0.95)
        
        # End of speech - slightly slower (wind down)
        if segment_index == total_segments - 1:
            rate = int(rate * 0.93)
        
        # Clamp to reasonable range
        return max(80, min(300, rate))
    
    def _apply_pitch_variation(self, base_pitch: int, segment_props: Dict,
                                emotion: str, segment_index: int, total_segments: int) -> int:
        """Apply natural pitch variation"""
        pitch = base_pitch
        
        # Apply segment modifier
        pitch += segment_props.get('pitch_modifier', 0)
        
        # Natural variation (±5)
        pitch += random.randint(-5, 5)
        
        # End of sentence - slight pitch drop (declarative)
        if segment_index == total_segments - 1:
            pitch -= 5
        
        # Questions - rising intonation at end
        if emotion == "curious" and segment_index == total_segments - 1:
            pitch += 10
        
        return max(50, min(200, pitch))
    
    def _apply_volume_variation(self, base_volume: float, segment_props: Dict,
                                 emotion: str) -> float:
        """Apply natural volume variation"""
        volume = base_volume * segment_props.get('volume_modifier', 1.0)
        
        # Natural variation (±0.05)
        volume += random.uniform(-0.03, 0.03)
        
        return max(0.2, min(1.0, volume))
    
    def _speak_with_rhythm_and_breathing(self, text: str, emotion: str, 
                                          audio_path: str, block: bool):
        """
        Speak text with natural rhythm, breathing, and human-like features.
        """
        # Apply human-like text preprocessing
        text = self._add_fillers(text, emotion)
        text = self._add_non_verbal(text, emotion)
        text = self._add_self_correction(text, emotion)
        
        # Preprocess text into segments with rhythm properties
        segments = self._preprocess_text_for_rhythm(text)
        total_segments = len(segments)
        
        if total_segments == 0:
            return
        
        # Save original properties
        orig_rate = self.engine.getProperty('rate')
        orig_volume = self.engine.getProperty('volume')
        try:
            orig_pitch = self.engine.getProperty('pitch')
            pitch_supported = orig_pitch is not None
        except:
            orig_pitch = self.pitch
            pitch_supported = True
        
        try:
            for i, (segment_text, props) in enumerate(segments):
                if not segment_text.strip():
                    continue
                
                # Check if this is a phrase/sentence end
                is_last_segment = (i == total_segments - 1)
                is_phrase_end = props.get('pause_after', 0) > 0
                
                # Apply pre-boundary lengthening
                boundary_props = self._apply_pre_boundary_lengthening(
                    segment_text, is_phrase_end, is_last_segment
                )
                
                # Calculate rhythm properties for this segment
                rate = self._apply_rhythm_variation(orig_rate, props, emotion, i, total_segments)
                pitch = self._apply_pitch_variation(orig_pitch, props, emotion, i, total_segments) if pitch_supported else None
                volume = self._apply_volume_variation(orig_volume, props, emotion)
                
                # Apply boundary lengthening
                rate = int(rate * boundary_props['rate_modifier'])
                if boundary_props['pitch_modifier']:
                    pitch = (pitch or orig_pitch) + boundary_props['pitch_modifier']
                
                # Apply properties
                self.engine.setProperty('rate', rate)
                self.engine.setProperty('volume', volume)
                if pitch_supported and pitch is not None:
                    try:
                        self.engine.setProperty('pitch', pitch)
                    except:
                        pass
                
                # Speak the segment with word-level emphasis (handles runAndWait internally)
                self._speak_with_emphasis(segment_text, emotion, pitch_supported)
                
                # Add pause after segment
                pause_duration = props.get('pause_after', 0)
                if pause_duration > 0:
                    time.sleep(pause_duration)
                
                # Add vocal fry at end of declarative sentences
                if boundary_props.get('vocal_fry', False):
                    self._add_vocal_fry(emotion)
                
                # Add breathing if indicated
                if props.get('breath', False):
                    self._add_breath(emotion)
                    
        finally:
            # Restore original properties
            self.engine.setProperty('rate', orig_rate)
            self.engine.setProperty('volume', orig_volume)
            try:
                self.engine.setProperty('pitch', orig_pitch)
            except:
                pass
    
    def _speak_with_emphasis(self, text: str, emotion: str, pitch_supported: bool):
        """
        Speak text with word-level emphasis.
        Applies emphasis to the whole segment based on detected emphasis words.
        """
        # Check for emphasis words in the text
        words = text.split()
        emphasis_props = {'rate_modifier': 1.0, 'pitch_modifier': 0, 'volume_modifier': 1.0}
        has_emphasis = False
        
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            for pattern, level in EMPHASIS_PATTERNS:
                if re.search(pattern, clean_word, re.IGNORECASE):
                    has_emphasis = True
                    if level == 'strong':
                        emphasis_props['rate_modifier'] = 0.85
                        emphasis_props['pitch_modifier'] = 8
                        emphasis_props['volume_modifier'] = 1.15
                    elif level == 'moderate':
                        emphasis_props['rate_modifier'] = 0.9
                        emphasis_props['pitch_modifier'] = 5
                        emphasis_props['volume_modifier'] = 1.1
                    elif level == 'light':
                        emphasis_props['rate_modifier'] = 0.95
                        emphasis_props['pitch_modifier'] = 3
                        emphasis_props['volume_modifier'] = 1.05
                    break
            if has_emphasis:
                break
        
        # Apply emphasis properties if needed
        if has_emphasis:
            orig_rate = self.engine.getProperty('rate')
            orig_volume = self.engine.getProperty('volume')
            try:
                orig_pitch = self.engine.getProperty('pitch')
            except:
                orig_pitch = self.pitch
            
            try:
                self.engine.setProperty('rate', int(orig_rate * emphasis_props['rate_modifier']))
                self.engine.setProperty('volume', min(1.0, orig_volume * emphasis_props['volume_modifier']))
                if pitch_supported:
                    try:
                        self.engine.setProperty('pitch', orig_pitch + emphasis_props['pitch_modifier'])
                    except:
                        pass
                
                self.engine.say(text)
                self.engine.runAndWait()
                
            finally:
                self.engine.setProperty('rate', orig_rate)
                self.engine.setProperty('volume', orig_volume)
                if pitch_supported:
                    try:
                        self.engine.setProperty('pitch', orig_pitch)
                    except:
                        pass
        else:
            # No emphasis, speak normally
            self.engine.say(text)
            self.engine.runAndWait()
    
    def _add_vocal_fry(self, emotion: str = "neutral"):
        """
        Add vocal fry (creaky voice) at end of phrases.
        Simulated with very low pitch and volume.
        """
        orig_rate = self.engine.getProperty('rate')
        orig_volume = self.engine.getProperty('volume')
        try:
            orig_pitch = self.engine.getProperty('pitch')
            pitch_supported = orig_pitch is not None
        except:
            orig_pitch = self.pitch
            pitch_supported = True
        
        try:
            # Very slow, low, quiet
            self.engine.setProperty('rate', int(orig_rate * 0.5))
            self.engine.setProperty('volume', 0.15)
            if pitch_supported:
                try:
                    self.engine.setProperty('pitch', max(50, orig_pitch - 30))
                except:
                    pass
            
            # Vocal fry sound
            self.engine.say("uh")
            self.engine.runAndWait()
            time.sleep(0.15)
            
        finally:
            self.engine.setProperty('rate', orig_rate)
            self.engine.setProperty('volume', orig_volume)
            if pitch_supported:
                try:
                    self.engine.setProperty('pitch', orig_pitch)
                except:
                    pass
    
    def _add_breath(self, emotion: str = "neutral"):
        """
        Add a natural breath pause.
        Since pyttsx3 doesn't support breath sounds directly,
        we simulate it with a pause and subtle voice change.
        """
        breath = self._generate_breath_pattern(emotion)
        
        # Save current properties
        orig_rate = self.engine.getProperty('rate')
        orig_volume = self.engine.getProperty('volume')
        try:
            orig_pitch = self.engine.getProperty('pitch')
            pitch_supported = orig_pitch is not None
        except:
            orig_pitch = self.pitch
            pitch_supported = True
        
        try:
            # Brief pause for breath
            time.sleep(breath["duration"] * 0.5)
            
            # Optional: emit a very soft "breath" sound using a whisper-like utterance
            # This is subtle - just a brief pause with modified voice
            self.engine.setProperty('rate', int(orig_rate * 0.5))
            self.engine.setProperty('volume', breath["volume"])
            if pitch_supported:
                try:
                    self.engine.setProperty('pitch', orig_pitch + breath["pitch"])
                except:
                    pass
            
            # Use a very short "h" sound or just pause
            self.engine.say("h")
            self.engine.runAndWait()
            
            time.sleep(breath["duration"] * 0.5)
            
        finally:
            # Restore
            self.engine.setProperty('rate', orig_rate)
            self.engine.setProperty('volume', orig_volume)
            try:
                self.engine.setProperty('pitch', orig_pitch)
            except:
                pass
    

    def speak_with_rhythm(self, text: str, emotion: str = None, use_rhythm: bool = True, use_breathing: bool = True) -> Optional[str]:
        """File-based rhythm+breathing render (local implementation). Returns wav path."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(self.temp_dir, f"aira_{int(time.time()*1000)}.wav")
        if use_rhythm and use_breathing:
            self._speak_with_rhythm_and_breathing(text, emotion or self.current_emotion, audio_path, True)
        else:
            self.engine.save_to_file(text, audio_path)
            self.engine.runAndWait()
        return audio_path

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

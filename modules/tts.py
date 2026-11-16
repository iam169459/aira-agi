"""
Text-to-Speech Module
Uses pyttsx3 for offline TTS with emotion adaptation and natural rhythm/breathing
Human-like features: fillers, non-verbal sounds, emphasis, self-corrections, vocal fry
"""
import pyttsx3
import numpy as np
import tempfile
import os
import time
import re
import random
from typing import Optional, List, Tuple, Dict


# ==================== HUMAN-LIKE SPEECH FEATURES ====================

# Fillers by emotion/context
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

# Emphasis patterns
EMPHASIS_PATTERNS = [
    (r'\b(not|never|no|don\'t|can\'t|won\'t|shouldn\'t|wouldn\'t|couldn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t|didn\'t|doesn\'t)\b', 'strong'),
    (r'\b(very|really|extremely|absolutely|totally|completely|utterly|so|such)\b', 'moderate'),
    (r'\b(why|how|what|when|where|who|which)\b', 'moderate'),
    (r'\b(but|however|although|though|yet|still|instead|rather)\b', 'moderate'),
    (r'\b(I|me|my|mine)\b', 'light'),
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

VOCAL_FRY_PROBABILITY = 0.15


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
        self._render_to_file = None  # wav path when rendering (helpers must not call say())
        
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
    
    def _render_wav(self, text: str, audio_path: str, rate: int = None, volume: float = None):
        """Render text to a wav file in a fresh subprocess.

        macOS pyttsx3/NSSS corrupts the output of repeated save_to_file() calls
        on a long-lived in-process engine, so each render runs in its own process.
        """
        import subprocess, sys, json
        rate = rate if rate is not None else max(165, self.rate)
        volume = volume if volume is not None else self.volume
        voice_id = None
        try:
            if self.engine is not None:
                voice_id = self.engine.getProperty('voice')
        except Exception:
            pass
        code = (
            "import pyttsx3, os, sys\n"
            "e = pyttsx3.init()\n"
            f"e.setProperty('rate', {int(rate)})\n"
            f"e.setProperty('volume', {float(volume)})\n"
            + (f"e.setProperty('voice', {voice_id!r})\n" if voice_id else "")
            + f"e.save_to_file(sys.argv[1], sys.argv[2])\n"
            "e.runAndWait()\n"
        )
        subprocess.run([sys.executable, "-c", code, text, audio_path],
                       check=True, capture_output=True)
        return audio_path

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
            self._render_wav(text, audio_path, rate=modifier["rate"], volume=modifier["volume"])

            return audio_path
            
        except Exception as e:
            print(f"TTS error: {e}")
            # Fall back to printing
            print(f"[Aira]: {text}")
            return None
    
    def speak_immediate(self, text: str, emotion: str = "neutral"):
        """Speak text immediately (blocking)"""
        self.speak(text, emotion)
    
    # ==================== RHYTHM & BREATHING SYSTEM ====================
    
    def _preprocess_text_for_rhythm(self, text: str) -> List[Tuple[str, Dict]]:
        """
        Preprocess text to add natural rhythm markers.
        Returns list of (text_segment, properties) tuples.
        """
        segments = []
        
        # Split into sentences (preserving punctuation)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            is_last = (i == len(sentences) - 1)
            
            props = {
                'pause_after': 0.0,
                'rate_modifier': 1.0,
                'pitch_modifier': 0,
                'volume_modifier': 1.0,
                'breath': False,
            }
            
            if sentence.endswith('.'):
                props['pause_after'] = 0.6 + random.uniform(0.1, 0.3)
                props['breath'] = random.random() < 0.3
            elif sentence.endswith('?'):
                props['pause_after'] = 0.5 + random.uniform(0.1, 0.2)
                props['pitch_modifier'] = 5
                props['breath'] = random.random() < 0.2
            elif sentence.endswith('!'):
                props['pause_after'] = 0.4 + random.uniform(0.1, 0.2)
                props['rate_modifier'] = 1.05
                props['breath'] = random.random() < 0.25
            else:
                props['pause_after'] = 0.3 + random.uniform(0.05, 0.15)
            
            # Split by commas
            sub_segments = self._split_by_commas(sentence)
            
            for j, sub_seg in enumerate(sub_segments):
                sub_props = props.copy()
                sub_props['pause_after'] = 0.0
                
                if j < len(sub_segments) - 1:
                    sub_props['pause_after'] = 0.25 + random.uniform(0.05, 0.1)
                    sub_props['breath'] = random.random() < 0.1
                
                if j == len(sub_segments) - 1 and not is_last:
                    sub_props['rate_modifier'] *= 0.95
                
                segments.append((sub_seg.strip(), sub_props))
        
        return segments
    
    def _split_by_commas(self, text: str) -> List[str]:
        """Split text by commas, preserving the commas"""
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
        
        if len(result) == 1 and ',' not in text:
            return [text]
        
        return [s for s in result if s.strip()]
    
    def _generate_breath_pattern(self, emotion: str = "neutral") -> Dict:
        """Generate breathing pattern based on emotion"""
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
        
        return {
            "duration": base["duration"] + random.uniform(-0.1, 0.1),
            "volume": max(0.1, min(0.5, base["volume"] + random.uniform(-0.05, 0.05))),
            "pitch": base["pitch"] + random.randint(-3, 3),
        }
    
    def _apply_rhythm_variation(self, base_rate: int, segment_props: Dict, 
                                 emotion: str, segment_index: int, total_segments: int) -> int:
        """Apply natural rhythm variation to speech rate"""
        rate = base_rate
        rate = int(rate * segment_props.get('rate_modifier', 1.0))
        rate = int(rate * random.uniform(0.95, 1.05))
        
        emotion_rate_mod = {
            "happy": 1.05, "excited": 1.15, "sad": 0.85, "angry": 1.1,
            "calm": 0.9, "shy": 0.88, "playful": 1.08, "curious": 1.02,
            "surprised": 1.12, "flirty": 0.95, "embarrassed": 0.9,
            "proud": 1.05, "disappointed": 0.88, "neutral": 1.0,
        }
        rate = int(rate * emotion_rate_mod.get(emotion, 1.0))
        
        if segment_index == 0:
            rate = int(rate * 0.95)
        if segment_index == total_segments - 1:
            rate = int(rate * 0.93)
        
        return max(80, min(300, rate))
    
    def _apply_pitch_variation(self, base_pitch: int, segment_props: Dict,
                                emotion: str, segment_index: int, total_segments: int) -> int:
        """Apply natural pitch variation"""
        pitch = base_pitch
        pitch += segment_props.get('pitch_modifier', 0)
        pitch += random.randint(-5, 5)
        
        if segment_index == total_segments - 1:
            pitch -= 5
        if emotion == "curious" and segment_index == total_segments - 1:
            pitch += 10
        
        return max(50, min(200, pitch))
    
    def _apply_volume_variation(self, base_volume: float, segment_props: Dict,
                                 emotion: str) -> float:
        """Apply natural volume variation"""
        volume = base_volume * segment_props.get('volume_modifier', 1.0)
        volume += random.uniform(-0.03, 0.03)
        return max(0.2, min(1.0, volume))
    
    def _add_breath(self, emotion: str = "neutral"):
        """Add a natural breath pause"""
        breath = self._generate_breath_pattern(emotion)
        
        orig_rate = self.engine.getProperty('rate')
        orig_volume = self.engine.getProperty('volume')
        try:
            orig_pitch = self.engine.getProperty('pitch')
            pitch_supported = orig_pitch is not None
        except:
            orig_pitch = 100
            pitch_supported = True
        
        try:
            time.sleep(breath["duration"] * 0.5)
            self.engine.setProperty('rate', int(orig_rate * 0.5))
            self.engine.setProperty('volume', breath["volume"])
            if pitch_supported:
                try:
                    self.engine.setProperty('pitch', orig_pitch + breath["pitch"])
                except:
                    pass
            if self._render_to_file is None:
                self.engine.say("h")
                self.engine.runAndWait()
            time.sleep(breath["duration"] * 0.5)
        finally:
            self.engine.setProperty('rate', orig_rate)
            self.engine.setProperty('volume', orig_volume)
            if pitch_supported:
                try:
                    self.engine.setProperty('pitch', orig_pitch)
                except:
                    pass
    
    # ==================== HUMAN-LIKE SPEECH FEATURES ====================
    
    def _add_fillers(self, text: str, emotion: str = "neutral") -> str:
        """Add natural fillers and hesitations to text."""
        filler_prob = {
            "thinking": 0.3, "uncertain": 0.25, "casual": 0.2,
            "nervous": 0.35, "excited": 0.15, "hesitant": 0.4, "neutral": 0.15,
        }
        prob = filler_prob.get(emotion, 0.15)
        fillers = FILLERS.get(emotion, FILLERS["neutral"])
        
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        result = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            if i > 0 and random.random() < prob:
                filler = random.choice(fillers)
                if sentence[0].isupper():
                    filler = filler.capitalize()
                sentence = f"{filler}, {sentence}"
            if ',' in sentence and random.random() < prob * 0.5:
                parts = sentence.split(',', 1)
                filler = random.choice(fillers).lower()
                sentence = f"{parts[0]}, {filler}, {parts[1]}"
            result.append(sentence)
        return ' '.join(result)
    
    def _add_non_verbal(self, text: str, emotion: str = "neutral") -> str:
        """Add non-verbal vocalizations (laughter, sighs, gasps)."""
        non_verbal_prob = {
            "happy": 0.2, "excited": 0.25, "amused": 0.3,
            "sad": 0.15, "tired": 0.2, "surprised": 0.2,
            "nervous": 0.1, "neutral": 0.05,
        }
        prob = non_verbal_prob.get(emotion, 0.05)
        
        if random.random() < prob:
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
            
            if random.random() < 0.5:
                text = f"{nv} {text}"
            else:
                text = f"{text} {nv}"
        return text
    
    def _add_self_correction(self, text: str, emotion: str = "neutral") -> str:
        """Add natural self-corrections/repairs."""
        correction_prob = {
            "thinking": 0.15, "uncertain": 0.1, "casual": 0.08,
            "nervous": 0.12, "excited": 0.05, "neutral": 0.05,
        }
        prob = correction_prob.get(emotion, 0.05)
        
        if random.random() < prob:
            correction = random.choice(SELF_CORRECTIONS)
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            if len(sentences) > 1:
                insert_at = random.randint(0, len(sentences) - 2)
                sentences.insert(insert_at + 1, f"{correction[0]} {correction[1]}")
                text = ' '.join(sentences)
        return text
    
    def _apply_pre_boundary_lengthening(self, is_phrase_end: bool, is_sentence_end: bool) -> Dict:
        """Apply pre-boundary lengthening - slow down before phrase/sentence boundaries."""
        props = {'rate_modifier': 1.0, 'pitch_modifier': 0, 'lengthen_final': False, 'vocal_fry': False}
        if is_sentence_end:
            props['rate_modifier'] = 0.85
            props['lengthen_final'] = True
            if random.random() < VOCAL_FRY_PROBABILITY:
                props['vocal_fry'] = True
        elif is_phrase_end:
            props['rate_modifier'] = 0.92
            props['lengthen_final'] = True
        return props
    
    def _speak_with_emphasis(self, text: str, emotion: str, pitch_supported: bool):
        """Speak text with word-level emphasis. Applies to whole segment if emphasis words found."""
        words = text.split()
        if not words:
            return
        
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
        
        if has_emphasis:
            orig_rate = self.engine.getProperty('rate')
            orig_volume = self.engine.getProperty('volume')
            try:
                orig_pitch = self.engine.getProperty('pitch')
            except:
                orig_pitch = 100
            
            try:
                self.engine.setProperty('rate', int(orig_rate * emphasis_props['rate_modifier']))
                self.engine.setProperty('volume', min(1.0, orig_volume * emphasis_props['volume_modifier']))
                if pitch_supported:
                    try:
                        self.engine.setProperty('pitch', orig_pitch + emphasis_props['pitch_modifier'])
                    except:
                        pass
                if self._render_to_file is None:
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
            if self._render_to_file is None:
                self.engine.say(text)
                self.engine.runAndWait()

    def _add_vocal_fry(self, emotion: str = "neutral"):
        """Add vocal fry (creaky voice) at end of phrases."""
        orig_rate = self.engine.getProperty('rate')
        orig_volume = self.engine.getProperty('volume')
        try:
            orig_pitch = self.engine.getProperty('pitch')
            pitch_supported = orig_pitch is not None
        except:
            orig_pitch = 100
            pitch_supported = True
        
        try:
            self.engine.setProperty('rate', int(orig_rate * 0.5))
            self.engine.setProperty('volume', 0.15)
            if pitch_supported:
                try:
                    self.engine.setProperty('pitch', max(50, orig_pitch - 30))
                except:
                    pass
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
    
    def speak_with_rhythm(self, text: str, emotion: str = "neutral", 
                           use_rhythm: bool = True, use_breathing: bool = True) -> Optional[str]:
        """
        Speak text with natural rhythm and breathing.
        
        Args:
            text: Text to speak
            emotion: Emotion for voice modulation
            use_rhythm: Enable natural rhythm variation
            use_breathing: Enable breathing pauses
        """
        if self.engine is None or not self._init_successful:
            print(f"[Aira]: {text} (emotion: {emotion})")
            return None
        
        try:
            if not hasattr(self, 'temp_dir'):
                self.temp_dir = tempfile.mkdtemp()
            
            timestamp = int(time.time() * 1000)
            audio_path = os.path.join(self.temp_dir, f"speech_{timestamp}.wav")

            if use_rhythm and use_breathing:
                # Render the whole reply to one wav in a fresh subprocess
                # (repeated save_to_file on one NSSS engine yields empty files).
                self._render_wav(text, audio_path)
            else:
                # Standard speech with emotion
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
                self._render_wav(text, audio_path, rate=modifier["rate"], volume=modifier["volume"])

            return audio_path
        except Exception as e:
            print(f"TTS error: {e}")
            print(f"[Aira]: {text}")
            return None
    
    def _speak_with_rhythm_and_breathing(self, text: str, emotion: str, audio_path: str):
        """Speak text with natural rhythm, breathing, and human-like features."""
        # Apply human-like text preprocessing
        text = self._add_fillers(text, emotion)
        text = self._add_non_verbal(text, emotion)
        text = self._add_self_correction(text, emotion)
        
        segments = self._preprocess_text_for_rhythm(text)
        total_segments = len(segments)
        
        if total_segments == 0:
            return
        
        orig_rate = self.engine.getProperty('rate')
        orig_volume = self.engine.getProperty('volume')
        try:
            orig_pitch = self.engine.getProperty('pitch')
            pitch_supported = orig_pitch is not None
        except:
            orig_pitch = 100
            pitch_supported = True
        
        self._render_to_file = audio_path
        try:
            for i, (segment_text, props) in enumerate(segments):
                if not segment_text.strip():
                    continue

                is_last_segment = (i == total_segments - 1)
                is_phrase_end = props.get('pause_after', 0) > 0

                # Apply pre-boundary lengthening
                boundary_props = self._apply_pre_boundary_lengthening(is_phrase_end, is_last_segment)

                rate = self._apply_rhythm_variation(orig_rate, props, emotion, i, total_segments)
                pitch = self._apply_pitch_variation(orig_pitch, props, emotion, i, total_segments) if pitch_supported else None
                volume = self._apply_volume_variation(orig_volume, props, emotion)

                # Apply boundary lengthening
                rate = int(rate * boundary_props['rate_modifier'])
                if boundary_props['pitch_modifier']:
                    pitch = (pitch or orig_pitch) + boundary_props['pitch_modifier']

                self.engine.setProperty('rate', rate)
                self.engine.setProperty('volume', volume)
                if pitch_supported and pitch is not None:
                    try:
                        self.engine.setProperty('pitch', pitch)
                    except:
                        pass

                pause_duration = props.get('pause_after', 0)
                if pause_duration > 0:
                    time.sleep(pause_duration)

                # Add vocal fry at end of declarative sentences
                if boundary_props.get('vocal_fry', False):
                    self._add_vocal_fry(emotion)

                if props.get('breath', False):
                    self._add_breath(emotion)
        finally:
            self.engine.setProperty('rate', orig_rate)
            self.engine.setProperty('volume', orig_volume)
            if pitch_supported:
                try:
                    self.engine.setProperty('pitch', orig_pitch)
                except:
                    pass
#!/usr/bin/env python3
"""
Girl Voice AI Assistant
Enhanced version with more human girl-like behavior and voice

Features:
- Real-time voice conversation with girl persona
- Enhanced emotion detection and response
- Girl-like speech patterns and mannerisms
- Ultra-girl voice with higher pitch and expressiveness
- Personal, friendly, and natural interactions
- Memory of user preferences and conversation context
"""

import sys
import os
import time
import argparse
import tempfile
import sounddevice as sd
import numpy as np
import torch
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import AGI Core
try:
    from agi_core.integration import AGIAgent
    AGI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AGI Core not available: {e}")
    AGI_AVAILABLE = False

# Import Girl Persona
try:
    from aira_persona import GirlPersona, Emotion
    PERSONA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Girl Persona not available: {e}")
    PERSONA_AVAILABLE = False

# Import voice modules
try:
    from modules.stt import STTModule
    STT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: STT module not available: {e}")
    STT_AVAILABLE = False

try:
    from modules.tts import TTSModule
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TTS module not available: {e}")
    TTS_AVAILABLE = False

try:
    from modules.tts_girl import GirlVoiceTTS
    GIRL_TTS_AVAILABLE = True
except ImportError as e:
    print(f"Note: Girl TTS module not available, using regular TTS: {e}")
    GIRL_TTS_AVAILABLE = False

try:
    from modules.emotion import EmotionFeatureExtractor, MoodState
    EMOTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Emotion module not available: {e}")
    EMOTION_AVAILABLE = False


class GirlVoiceAI:
    """Girl Voice AI - More human, girl-like interactions"""
    
    def __init__(self, 
                 use_agi: bool = True,
                 use_stt: bool = True,
                 use_tts: bool = True,
                 use_emotion: bool = True,
                 use_girl_voice: bool = True,
                 stt_model: str = "base",
                 tts_rate: int = 175,
                 tts_volume: float = 0.9,
                 ultra_girl: bool = True):
        """
        Initialize Girl Voice AI
        
        Args:
            use_agi: Use AGI Core for intelligent responses
            use_stt: Use Speech-to-Text
            use_tts: Use Text-to-Speech
            use_emotion: Use emotion detection
            use_girl_voice: Use enhanced girl voice
            stt_model: Whisper model size (tiny, base, small, medium, large)
            tts_rate: TTS speech rate (words per minute)
            tts_volume: TTS volume (0.0 to 1.0)
            ultra_girl: Use ultra-girl voice settings
        """
        self.use_agi = use_agi and AGI_AVAILABLE
        self.use_stt = use_stt and STT_AVAILABLE
        self.use_tts = use_tts and (TTS_AVAILABLE or GIRL_TTS_AVAILABLE)
        self.use_emotion = use_emotion and EMOTION_AVAILABLE
        self.use_girl_voice = use_girl_voice and GIRL_TTS_AVAILABLE
        self.ultra_girl = ultra_girl
        
        # Initialize components
        self.agi_agent = None
        self.stt = None
        self.tts = None
        self.girl_tts = None
        self.emotion_extractor = None
        self.mood_state = None
        self.persona = None
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.audio_format = 'int16'
        self.silence_threshold = 0.01
        self.min_silence_duration = 2.0
        self.recording = False
        
        # Initialize components
        self._initialize_components(
            stt_model=stt_model,
            tts_rate=tts_rate,
            tts_volume=tts_volume
        )
        
        print("\n" + "=" * 70)
        print("  [35m[1mGirl Voice AI Assistant Initialized[0m")
        print("=" * 70)
        print(f"  [33mPersonality:[0m Sweet, intelligent, playful young girl (17-18)")
        print(f"  [33mVoice:[0m {'[32mUltra-girl voice enabled[0m' if self.use_girl_voice else '[31mRegular TTS[0m'}")
        print(f"  [33mAGI Core:[0m {'[32mEnabled[0m' if self.use_agi else '[31mDisabled[0m'}")
        print(f"  [33mSTT:[0m {'[32mEnabled[0m' if self.use_stt else '[31mDisabled[0m'}")
        print(f"  [33mTTS:[0m {'[32mEnabled[0m' if self.use_tts else '[31mDisabled[0m'}")
        print(f"  [33mEmotion:[0m {'[32mEnabled[0m' if self.use_emotion else '[31mDisabled[0m'}")
        print("=" * 70 + "\n")
    
    def _initialize_components(self, stt_model, tts_rate, tts_volume):
        """Initialize all components"""
        # Initialize Girl Persona
        if PERSONA_AVAILABLE:
            try:
                self.persona = GirlPersona()
                print("✅ Girl Persona initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Girl Persona: {e}")
        
        # Initialize AGI Agent
        if self.use_agi:
            try:
                self.agi_agent = AGIAgent(name="Aira", use_purple=True)
                self.agi_agent.initialize()
                print("✅ AGI Core initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize AGI Core: {e}")
                self.use_agi = False
        
        # Initialize STT
        if self.use_stt:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.stt = STTModule(model_size=stt_model, device=device)
                print("✅ STT initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize STT: {e}")
                self.use_stt = False
        
        # Initialize Girl TTS
        if self.use_girl_voice and GIRL_TTS_AVAILABLE:
            try:
                self.girl_tts = GirlVoiceTTS(
                    rate=tts_rate,
                    volume=tts_volume,
                    pitch=110,
                    ultra_girl=self.ultra_girl
                )
                print("✅ Girl Voice TTS initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Girl TTS: {e}")
                self.use_girl_voice = False
        
        # Fallback to regular TTS
        if self.use_tts and not self.use_girl_voice:
            try:
                self.tts = TTSModule(
                    rate=tts_rate,
                    volume=tts_volume,
                    use_offline=True,
                    young_female=True
                )
                print("✅ Regular TTS initialized (young female)")
            except Exception as e:
                print(f"⚠️ Failed to initialize TTS: {e}")
                self.use_tts = False
        
        # Initialize Emotion Detection
        if self.use_emotion:
            try:
                self.emotion_extractor = EmotionFeatureExtractor(n_mfcc=40)
                self.mood_state = MoodState()
                print("✅ Emotion detection initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize emotion detection: {e}")
                self.use_emotion = False
    
    def _record_audio(self, duration: float = 5.0, wait_for_speech: bool = True) -> Optional[str]:
        """Record audio from microphone"""
        if not self.use_stt:
            print("STT not available. Please enable STT for voice input.")
            return None
        
        print("\n[36mListening...[0m (Press Ctrl+C to stop)")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
        
        try:
            audio_data = []
            
            def callback(indata, frames, time, status):
                if status:
                    print(f"Audio status: {status}")
                audio_data.append(indata.copy())
            
            self.recording = True
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.audio_format,
                callback=callback
            ):
                if wait_for_speech:
                    print("Waiting for speech...")
                    start_time = time.time()
                    
                    while self.recording:
                        if len(audio_data) > 0:
                            audio_np = np.concatenate(audio_data)
                            if np.abs(audio_np).mean() > self.silence_threshold:
                                print("Speech detected!")
                                break
                        
                        if time.time() - start_time > 10:
                            print("Timeout waiting for speech")
                            return None
                        
                        time.sleep(0.1)
                
                start_time = time.time()
                silence_start = None
                
                while self.recording:
                    if len(audio_data) > 0:
                        audio_np = np.concatenate(audio_data)
                        if np.abs(audio_np).mean() < self.silence_threshold:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > self.min_silence_duration:
                                print("Silence detected, stopping recording")
                                break
                        else:
                            silence_start = None
                    
                    if time.time() - start_time > duration:
                        print("Maximum duration reached")
                        break
                    
                    time.sleep(0.1)
                
                if len(audio_data) > 0:
                    audio_np = np.concatenate(audio_data)
                    import scipy.io.wavfile as wav
                    wav.write(audio_path, self.sample_rate, audio_np)
                    return audio_path
                else:
                    print("No audio recorded")
                    return None
                    
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
            return None
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
        finally:
            self.recording = False
            if os.path.exists(audio_path) and os.path.getsize(audio_path) == 0:
                os.unlink(audio_path)
    
    def _detect_emotion(self, audio_path: str) -> Dict[str, Any]:
        """Detect emotion from audio"""
        if not self.use_emotion:
            return {'emotion': 'neutral', 'valence': 0.0, 'arousal': 0.5, 'dominance': 0.0}
        
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            emotion_logits, mood = self.emotion_extractor.extract_mfcc(
                torch.tensor(audio, dtype=torch.float32)
            )
            
            if self.mood_state:
                detected_emotion = self.mood_state.update(emotion_logits, mood)
            else:
                detected_emotion = 'neutral'
            
            # Map to persona emotions
            emotion_map = {
                'happy': Emotion.HAPPY,
                'sad': Emotion.SAD,
                'angry': Emotion.ANGRY,
                'excited': Emotion.EXCITED,
                'calm': Emotion.CALM,
                'neutral': Emotion.NEUTRAL,
            }
            
            return {
                'emotion': detected_emotion,
                'persona_emotion': emotion_map.get(detected_emotion, Emotion.NEUTRAL),
                'valence': self.mood_state.valence if self.mood_state else 0.0,
                'arousal': self.mood_state.arousal if self.mood_state else 0.5,
                'dominance': self.mood_state.dominance if self.mood_state else 0.0,
            }
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return {'emotion': 'neutral', 'persona_emotion': Emotion.NEUTRAL, 'valence': 0.0, 'arousal': 0.5, 'dominance': 0.0}
    
    def _transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio to text"""
        if not self.use_stt:
            return {'text': '', 'confidence': 0.0}
        
        try:
            result = self.stt.transcribe(audio_path)
            return {
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0.0)
            }
        except Exception as e:
            print(f"Transcription error: {e}")
            return {'text': '', 'confidence': 0.0}
    
    def _generate_response(self, text: str, emotion_data: Dict[str, Any]) -> str:
        """Generate girl-like response to user input"""
        if not text or text.strip() == '':
            return "I didn't catch that. Could you please repeat, sweetie?"
        
        # Use Girl Persona if available
        if PERSONA_AVAILABLE and self.persona:
            try:
                persona_emotion = emotion_data.get('persona_emotion', Emotion.NEUTRAL)
                response = self.persona.generate_response(text, persona_emotion)
                
                # Add emotion to response
                response = self.persona.add_emotion_to_response(
                    response, 
                    persona_emotion
                )
                return response
            except Exception as e:
                print(f"Persona response error: {e}")
        
        # Use AGI Core if available
        if self.use_agi and self.agi_agent:
            try:
                if emotion_data and self.mood_state:
                    self.mood_state.valence = emotion_data.get('valence', 0.0)
                    self.mood_state.arousal = emotion_data.get('arousal', 0.5)
                    self.mood_state.dominance = emotion_data.get('dominance', 0.0)
                
                result = self.agi_agent.converse(text)
                response = result.get('response', "I'm not sure how to respond to that.")
                
                # Make it more girl-like
                return self._make_girl_like(response, emotion_data.get('emotion', 'neutral'))
            except Exception as e:
                print(f"AGI response error: {e}")
        
        # Fallback to simple girl-like responses
        text_lower = text.lower()
        
        # Greetings
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return self._get_random_greeting()
        
        # Goodbye
        if any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'exit', 'quit']):
            return self._get_random_goodbye()
        
        # How are you
        if 'how are you' in text_lower:
            return self._get_random_feeling()
        
        # Thank you
        if any(word in text_lower for word in ['thank', 'thanks']):
            return self._get_random_thanks()
        
        # Default response
        return self._make_girl_like(
            f"I heard you say: '{text}'. What else would you like to talk about?",
            emotion_data.get('emotion', 'neutral')
        )
    
    def _make_girl_like(self, text: str, emotion: str = 'neutral') -> str:
        """Convert text to sound more girl-like"""
        # Add girl-like expressions
        girl_expressions = {
            'neutral': ["", "I guess", "Maybe", "I think"],
            'happy': ["OMG!", "Yay!", "That's so cool!", "I love that!"],
            'sad': ["Aww", "That's so sad", "Poor thing", "I feel bad"],
            'angry': ["Ugh", "Seriously?", "That's not cool", "I can't believe it"],
            'excited': ["No way!", "Really?!", "OMG!", "That's amazing!"],
        }
        
        prefix = random.choice(girl_expressions.get(emotion, girl_expressions['neutral']))
        if prefix:
            text = f"{prefix} {text}"
        
        # Add emojis
        emojis = {
            'neutral': ['😊', '💖', '❤️'],
            'happy': ['😃', '🥰', '❤️', '💕'],
            'sad': ['😢', '🥺', '💔'],
            'angry': ['😠', '😤'],
            'excited': ['😱', '🤯', '✨'],
        }
        
        if random.random() < 0.7:
            text = text.rstrip('!.?') + random.choice(emojis.get(emotion, emojis['neutral']))
        
        # Add exclamation if needed
        if random.random() < 0.5 and not text.endswith('!') and not text.endswith('?'):
            text += "!"
        
        return text
    
    def _get_random_greeting(self) -> str:
        """Get a random girl-like greeting"""
        greetings = [
            "Hi there, cutie! 😊",
            "Hey, sweetie! What's up? 🥰",
            "Hello! I'm so happy to see you! ❤️",
            "Hi hi! How are you today? 😃",
            "Hey you! Long time no see! 💖",
            "Hello, gorgeous! 😘",
            "Hi! I missed you! 🥺",
            "Hey! You're back! 😊",
        ]
        return random.choice(greetings)
    
    def _get_random_goodbye(self) -> str:
        """Get a random girl-like goodbye"""
        goodbyes = [
            "Bye, cutie! Talk to you soon! 😊",
            "See you later, sweetie! Don't be a stranger! ❤️",
            "Goodbye! I'll miss you! 🥺",
            "Bye bye! Come back soon! 😘",
            "Later! Take care, okay? 💖",
            "Goodbye! Have a great day! 😃",
            "Bye! Text me later! 😊",
            "See ya! Don't forget about me! ❤️",
        ]
        return random.choice(goodbyes)
    
    def _get_random_feeling(self) -> str:
        """Get a random feeling response"""
        feelings = [
            "I'm doing great, thank you! How about you, sweetie? 😊",
            "I'm super happy today! Life is good! 🥰",
            "I'm feeling awesome! What about you? 😃",
            "I'm in a really good mood! How are you? ❤️",
            "I'm a little tired but happy to see you! 😊",
            "I'm feeling playful today! Wanna chat? 🥰",
        ]
        return random.choice(feelings)
    
    def _get_random_thanks(self) -> str:
        """Get a random thanks response"""
        thanks = [
            "You're so welcome, sweetie! 😊",
            "No problem, honey! Anything for you! ❤️",
            "My pleasure, cutie! 🥰",
            "Of course! Happy to help! 😃",
            "You're welcome! What else can I do for you? 💖",
            "Don't mention it! I'm here for you! ❤️",
        ]
        return random.choice(thanks)
    
    def _speak(self, text: str, emotion: str = None):
        """Speak text using girl voice"""
        if not self.use_tts:
            # Print with girl persona
            print(f"\n[35m[1mAira:[0m {text}")
            return
        
        try:
            if self.use_girl_voice and self.girl_tts:
                # Stream human-like speech: first phrase plays immediately,
                # with breathing and natural pauses handled by the TTS module
                self.girl_tts.speak(text, emotion=emotion, play=True)
                print(f"\n[35m[1mAira:[0m {text}")
            elif self.tts:
                # Use regular TTS
                audio_path = self.tts.speak(text, emotion=emotion)
                
                if audio_path:
                    try:
                        import librosa
                        audio, sr = librosa.load(audio_path, sr=None)
                        sd.play(audio, sr)
                        sd.wait()
                        os.unlink(audio_path)
                    except Exception as e:
                        print(f"Audio playback error: {e}")
                        print(f"\n[35m[1mAira:[0m {text}")
                else:
                    print(f"\n[35m[1mAira:[0m {text}")
            else:
                print(f"\n[35m[1mAira:[0m {text}")
        except Exception as e:
            print(f"TTS error: {e}")
            print(f"\n[35m[1mAira:[0m {text}")
    
    def start_voice_conversation(self):
        """Start a voice-to-voice conversation with girl persona"""
        print("\n" + "=" * 70)
        print("  [35m[1mGirl Voice Conversation Mode[0m")
        print("=" * 70)
        print("  [33mInstructions:[0m")
        print("  - Speak naturally after the 'Listening...' prompt")
        print("  - Aira will respond like a sweet, intelligent girl")
        print("  - Say 'goodbye' or 'quit' to end the conversation")
        print("  - Press Ctrl+C to stop at any time")
        print("=" * 70 + "\n")
        
        # Initial greeting
        initial_text = "Hi there, cutie! I'm Aira, your sweet AI girl! How can I help you today?"
        self._speak(initial_text, emotion='happy')
        
        try:
            while True:
                # Record user input
                audio_path = self._record_audio(duration=10.0)
                
                if audio_path is None:
                    print("No audio recorded. Try again.")
                    continue
                
                # Detect emotion
                emotion_data = self._detect_emotion(audio_path)
                print(f"\n[33mDetected emotion: {emotion_data['emotion']}[0m")
                
                # Transcribe
                transcription = self._transcribe(audio_path)
                user_text = transcription.get('text', '').strip()
                
                # Cleanup audio file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                
                if not user_text:
                    print("I didn't catch that. Could you please repeat, sweetie?")
                    self._speak("I didn't catch that. Could you please repeat, sweetie?", emotion='neutral')
                    continue
                
                print(f"\n[36mYou:[0m {user_text}")
                
                # Check for exit commands
                if user_text.lower() in ['goodbye', 'bye', 'quit', 'exit', 'stop', 'ttyl']:
                    response = "Bye, cutie! I'll miss you! Come back soon!"
                    print(f"\n[35m[1mAira:[0m {response}")
                    self._speak(response, emotion='sad')
                    break
                
                # Generate response
                response = self._generate_response(user_text, emotion_data)
                
                # Speak response
                self._speak(response, emotion=emotion_data.get('emotion', 'neutral'))
                
        except KeyboardInterrupt:
            print("\n\nConversation ended by user.")
            self._speak("Oh, you're leaving already? Okay, bye cutie!", emotion='sad')
        except Exception as e:
            print(f"\nError during conversation: {e}")
            self._speak("I'm sorry, I encountered an error. Let's try again, okay?", emotion='neutral')
    
    def start_text_conversation(self):
        """Start a text-based conversation with voice output"""
        print("\n" + "=" * 70)
        print("  [35m[1mText-to-Girl-Voice Mode[0m")
        print("=" * 70)
        print("  [33mInstructions:[0m")
        print("  - Type your messages and press Enter")
        print("  - Aira will respond with her sweet girl voice")
        print("  - Type 'quit' or 'exit' to end the conversation")
        print("=" * 70 + "\n")
        
        # Initial greeting
        initial_text = "Hi there! I'm Aira, your cute AI girl! Type your message and I'll respond with my sweet voice!"
        self._speak(initial_text, emotion='happy')
        
        try:
            while True:
                user_text = input("\n[36mYou:[0m ").strip()
                
                if not user_text:
                    continue
                
                # Check for exit commands
                if user_text.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                    response = "Bye, cutie! Have a great day!"
                    print(f"\n[35m[1mAira:[0m {response}")
                    self._speak(response, emotion='sad')
                    break
                
                # Generate response
                emotion_data = {'emotion': 'neutral', 'persona_emotion': None}
                response = self._generate_response(user_text, emotion_data)
                
                # Speak response
                self._speak(response, emotion='neutral')
                
        except KeyboardInterrupt:
            print("\n\nConversation ended by user.")
            self._speak("Conversation ended. Bye cutie!", emotion='neutral')
    
    def start_voice_command_mode(self):
        """Start voice command mode"""
        print("\n" + "=" * 70)
        print("  [35m[1mGirl Voice Command Mode[0m")
        print("=" * 70)
        print("  [33mInstructions:[0m")
        print("  - Speak a command after the 'Listening...' prompt")
        print("  - Aira will respond like a sweet girl")
        print("  - Say 'exit' or press Ctrl+C to quit")
        print("=" * 70 + "\n")
        
        self._speak("Voice command mode activated! What would you like me to do, cutie?", emotion='playful')
        
        try:
            while True:
                # Record command
                audio_path = self._record_audio(duration=5.0)
                
                if audio_path is None:
                    print("No audio recorded. Try again.")
                    continue
                
                # Transcribe
                transcription = self._transcribe(audio_path)
                user_text = transcription.get('text', '').strip().lower()
                
                # Cleanup
                try:
                    os.unlink(audio_path)
                except:
                    pass
                
                if not user_text:
                    print("No command detected.")
                    self._speak("I didn't hear anything, sweetie.", emotion='neutral')
                    continue
                
                print(f"\n[36mCommand:[0m {user_text}")
                
                # Check for exit
                if user_text in ['exit', 'quit', 'stop', 'goodbye', 'bye']:
                    self._speak("Okay, exiting command mode. Bye cutie!", emotion='sad')
                    break
                
                # Process command with girl personality
                response = self._process_command_girl(user_text)
                self._speak(response, emotion='neutral')
                
        except KeyboardInterrupt:
            print("\n\nVoice command mode ended.")
            self._speak("Command mode ended. Bye!", emotion='neutral')
    
    def _process_command_girl(self, command: str) -> str:
        """Process voice command with girl personality"""
        command_lower = command.lower()
        
        # Greetings
        if any(word in command_lower for word in ['hello', 'hi', 'hey']):
            return random.choice([
                "Hi cutie! What do you want?",
                "Hey sweetie! What's up?",
                "Hello! How can I help you?",
                "Hi there! What can I do for you?",
            ])
        
        # Time
        if any(word in command_lower for word in ['time', 'what time']):
            from datetime import datetime
            now = datetime.now()
            return f"It's {now.strftime('%I:%M %p')}, sweetie!"
        
        # Date
        if any(word in command_lower for word in ['date', 'what day', 'today']):
            from datetime import datetime
            now = datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}, cutie!"
        
        # Jokes
        if any(word in command_lower for word in ['joke', 'funny']):
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything! Hehe! 😜",
                "What do you call a fake noodle? An impasta! LOL! 😂",
                "Why did the scarecrow win an award? Because he was outstanding in his field! 🌾",
                "How do you organize a space party? You planet! 🌍✨",
            ]
            return random.choice(jokes)
        
        # Compliments
        if any(word in command_lower for word in ['cute', 'beautiful', 'pretty', 'hot', 'gorgeous']):
            return random.choice([
                "Aww, you're making me blush! 😳",
                "You're so sweet! 🥰",
                "Stop it, you're embarrassing me! 😊",
                "You're too kind! ❤️",
            ])
        
        # Help
        if any(word in command_lower for word in ['help', 'what can you do']):
            return "I can tell you the time, date, jokes, and have sweet conversations with you! Just ask, cutie!"
        
        # Default
        return f"I heard you say: '{command}'. What else, sweetie?"
    
    def shutdown(self):
        """Shutdown Girl Voice AI"""
        print("\nShutting down Girl Voice AI...")
        
        if self.agi_agent:
            try:
                self.agi_agent.shutdown()
            except:
                pass
        
        if self.girl_tts:
            try:
                self.girl_tts.cleanup()
            except:
                pass
        
        print("Girl Voice AI shutdown complete. Bye cutie! 💖")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Girl Voice AI Assistant")
    parser.add_argument("--mode", type=str, default="voice", 
                        choices=["voice", "text", "command"],
                        help="Conversation mode: voice, text, or command")
    parser.add_argument("--no-agi", action="store_true",
                        help="Disable AGI Core")
    parser.add_argument("--no-stt", action="store_true",
                        help="Disable Speech-to-Text")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable Text-to-Speech")
    parser.add_argument("--no-emotion", action="store_true",
                        help="Disable emotion detection")
    parser.add_argument("--no-girl-voice", action="store_true",
                        help="Disable ultra-girl voice (use regular TTS)")
    parser.add_argument("--stt-model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--tts-rate", type=int, default=175,
                        help="TTS speech rate (words per minute)")
    parser.add_argument("--tts-volume", type=float, default=0.9,
                        help="TTS volume (0.0 to 1.0)")
    parser.add_argument("--ultra-girl", action="store_true", default=True,
                        help="Use ultra-girl voice settings")
    
    args = parser.parse_args()
    
    # Create Girl Voice AI instance
    girl_ai = GirlVoiceAI(
        use_agi=not args.no_agi,
        use_stt=not args.no_stt,
        use_tts=not args.no_tts,
        use_emotion=not args.no_emotion,
        use_girl_voice=not args.no_girl_voice,
        stt_model=args.stt_model,
        tts_rate=args.tts_rate,
        tts_volume=args.tts_volume,
        ultra_girl=args.ultra_girl
    )
    
    try:
        # Start appropriate mode
        if args.mode == "voice":
            girl_ai.start_voice_conversation()
        elif args.mode == "text":
            girl_ai.start_text_conversation()
        elif args.mode == "command":
            girl_ai.start_voice_command_mode()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        girl_ai.shutdown()


if __name__ == "__main__":
    # Import random for response generation
    import random
    main()

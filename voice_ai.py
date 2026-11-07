#!/usr/bin/env python3
"""
Voice-to-Voice AI Assistant
Integrates Speech-to-Text (STT) and Text-to-Speech (TTS) for natural voice conversations
with the Aira AGI system.

Features:
- Real-time voice conversation
- Emotion detection from voice
- Emotion-aware responses
- Integration with AGI Core for intelligent responses
- Young female voice preference for TTS
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
    from modules.emotion import EmotionFeatureExtractor, MoodState
    EMOTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Emotion module not available: {e}")
    EMOTION_AVAILABLE = False


class VoiceAI:
    """Main voice AI class for voice-to-voice conversations"""
    
    def __init__(self, 
                 use_agi: bool = True,
                 use_stt: bool = True,
                 use_tts: bool = True,
                 use_emotion: bool = True,
                 stt_model: str = "base",
                 tts_rate: int = 170,
                 tts_volume: float = 0.9,
                 young_female: bool = True):
        """
        Initialize Voice AI
        
        Args:
            use_agi: Use AGI Core for intelligent responses
            use_stt: Use Speech-to-Text
            use_tts: Use Text-to-Speech
            use_emotion: Use emotion detection
            stt_model: Whisper model size (tiny, base, small, medium, large)
            tts_rate: TTS speech rate (words per minute)
            tts_volume: TTS volume (0.0 to 1.0)
            young_female: Prefer young female voice for TTS
        """
        self.use_agi = use_agi and AGI_AVAILABLE
        self.use_stt = use_stt and STT_AVAILABLE
        self.use_tts = use_tts and TTS_AVAILABLE
        self.use_emotion = use_emotion and EMOTION_AVAILABLE
        
        # Initialize components
        self.agi_agent = None
        self.stt = None
        self.tts = None
        self.emotion_extractor = None
        self.mood_state = None
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.audio_format = 'int16'
        self.silence_threshold = 0.01
        self.min_silence_duration = 2.0  # seconds
        self.recording = False
        
        # Initialize components
        self._initialize_components(
            stt_model=stt_model,
            tts_rate=tts_rate,
            tts_volume=tts_volume,
            young_female=young_female
        )
        
        print("\n" + "=" * 70)
        print("  [35mVoice AI Assistant Initialized[0m")
        print("=" * 70)
        print(f"  AGI Core: {'[32mEnabled[0m' if self.use_agi else '[31mDisabled[0m'}")
        print(f"  STT: {'[32mEnabled[0m' if self.use_stt else '[31mDisabled[0m'}")
        print(f"  TTS: {'[32mEnabled[0m' if self.use_tts else '[31mDisabled[0m'}")
        print(f"  Emotion: {'[32mEnabled[0m' if self.use_emotion else '[31mDisabled[0m'}")
        print("=" * 70 + "\n")
    
    def _initialize_components(self, stt_model, tts_rate, tts_volume, young_female):
        """Initialize all components"""
        # Initialize AGI Agent
        if self.use_agi:
            try:
                self.agi_agent = AGIAgent(name="Aira", use_purple=True)
                self.agi_agent.initialize()
                print("AGI Core initialized successfully")
            except Exception as e:
                print(f"Failed to initialize AGI Core: {e}")
                self.use_agi = False
        
        # Initialize STT
        if self.use_stt:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.stt = STTModule(model_size=stt_model, device=device)
                print(f"STT initialized with {stt_model} model")
            except Exception as e:
                print(f"Failed to initialize STT: {e}")
                self.use_stt = False
        
        # Initialize TTS
        if self.use_tts:
            try:
                self.tts = TTSModule(
                    rate=tts_rate,
                    volume=tts_volume,
                    use_offline=True,
                    young_female=young_female
                )
                print(f"TTS initialized (rate: {tts_rate}, volume: {tts_volume})")
            except Exception as e:
                print(f"Failed to initialize TTS: {e}")
                self.use_tts = False
        
        # Initialize Emotion Detection
        if self.use_emotion:
            try:
                self.emotion_extractor = EmotionFeatureExtractor(n_mfcc=40)
                self.mood_state = MoodState()
                print("Emotion detection initialized")
            except Exception as e:
                print(f"Failed to initialize emotion detection: {e}")
                self.use_emotion = False
    
    def _record_audio(self, duration: float = 5.0, wait_for_speech: bool = True) -> Optional[str]:
        """
        Record audio from microphone
        
        Args:
            duration: Maximum recording duration in seconds
            wait_for_speech: Wait for speech to start recording
        
        Returns:
            Path to recorded audio file, or None if failed
        """
        if not self.use_stt:
            print("STT not available. Please enable STT for voice input.")
            return None
        
        print("\n[36mListening...[0m (Press Ctrl+C to stop)")
        
        # Create temp file for audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
        
        try:
            # Recording callback
            audio_data = []
            
            def callback(indata, frames, time, status):
                if status:
                    print(f"Audio status: {status}")
                audio_data.append(indata.copy())
            
            # Start recording
            self.recording = True
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.audio_format,
                callback=callback
            ):
                # Wait for user to speak
                if wait_for_speech:
                    print("Waiting for speech...")
                    start_time = time.time()
                    
                    while self.recording:
                        # Check if we have audio data
                        if len(audio_data) > 0:
                            # Check audio level
                            audio_np = np.concatenate(audio_data)
                            if np.abs(audio_np).mean() > self.silence_threshold:
                                print("Speech detected!")
                                break
                        
                        # Timeout check
                        if time.time() - start_time > 10:
                            print("Timeout waiting for speech")
                            return None
                        
                        time.sleep(0.1)
                
                # Record for duration or until silence
                start_time = time.time()
                silence_start = None
                
                while self.recording:
                    # Check for silence
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
                    
                    # Check duration
                    if time.time() - start_time > duration:
                        print("Maximum duration reached")
                        break
                    
                    time.sleep(0.1)
                
                # Save audio to file
                if len(audio_data) > 0:
                    audio_np = np.concatenate(audio_data)
                    # Save as WAV file
                    import scipy.io.wavfile as wav
                    wav.write(audio_path, self.sample_rate, audio_np)
                    print(f"Audio saved to {audio_path}")
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
            # Cleanup temp file if it exists but is empty
            if os.path.exists(audio_path) and os.path.getsize(audio_path) == 0:
                os.unlink(audio_path)
    
    def _detect_emotion(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect emotion from audio
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Emotion detection results
        """
        if not self.use_emotion:
            return {
                'emotion': 'neutral',
                'valence': 0.0,
                'arousal': 0.5,
                'dominance': 0.0,
                'confidence': 0.0
            }
        
        try:
            # Load audio
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            emotion_logits, mood = self.emotion_extractor.extract_mfcc(
                torch.tensor(audio, dtype=torch.float32)
            )
            
            # Update mood state
            if self.mood_state:
                detected_emotion = self.mood_state.update(emotion_logits, mood)
            else:
                detected_emotion = 'neutral'
            
            return {
                'emotion': detected_emotion,
                'valence': self.mood_state.valence if self.mood_state else 0.0,
                'arousal': self.mood_state.arousal if self.mood_state else 0.5,
                'dominance': self.mood_state.dominance if self.mood_state else 0.0,
                'confidence': 0.8
            }
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return {
                'emotion': 'neutral',
                'valence': 0.0,
                'arousal': 0.5,
                'dominance': 0.0,
                'confidence': 0.0
            }
    
    def _transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Transcription results
        """
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
    
    def _generate_response(self, text: str, emotion: Dict[str, Any]) -> str:
        """
        Generate response to user input
        
        Args:
            text: User input text
            emotion: Detected emotion
        
        Returns:
            AI response text
        """
        if not text or text.strip() == '':
            return "I didn't catch that. Could you please repeat?"
        
        # Use AGI Core if available
        if self.use_agi and self.agi_agent:
            try:
                # Set mood based on emotion
                if emotion and self.mood_state:
                    self.mood_state.valence = emotion.get('valence', 0.0)
                    self.mood_state.arousal = emotion.get('arousal', 0.5)
                    self.mood_state.dominance = emotion.get('dominance', 0.0)
                
                result = self.agi_agent.converse(text)
                return result.get('response', "I'm not sure how to respond to that.")
            except Exception as e:
                print(f"AGI response error: {e}")
        
        # Fallback to simple responses
        text_lower = text.lower()
        
        # Greetings
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! How can I help you today?"
        
        # Goodbye
        if any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'exit', 'quit']):
            return "Goodbye! Have a great day!"
        
        # How are you
        if 'how are you' in text_lower:
            return "I'm doing well, thank you for asking! How about you?"
        
        # Thank you
        if any(word in text_lower for word in ['thank', 'thanks']):
            return "You're welcome! Is there anything else I can help with?"
        
        # Default response
        return f"I heard you say: '{text}'. How can I assist you with that?"
    
    def _speak(self, text: str, emotion: str = None):
        """
        Speak text using TTS
        
        Args:
            text: Text to speak
            emotion: Emotion for voice modulation
        """
        if not self.use_tts or not self.tts:
            print(f"\n[34mAira:[0m {text}")
            return
        
        try:
            if emotion is None:
                emotion = 'neutral'
            
            audio_path = self.tts.speak(text, emotion=emotion)
            
            if audio_path:
                # Play the audio
                try:
                    audio, sr = self._load_audio(audio_path)
                    sd.play(audio, sr)
                    sd.wait()
                    # Cleanup
                    os.unlink(audio_path)
                except Exception as e:
                    print(f"Audio playback error: {e}")
                    print(f"\n[34mAira:[0m {text}")
            else:
                print(f"\n[34mAira:[0m {text}")
        except Exception as e:
            print(f"TTS error: {e}")
            print(f"\n[34mAira:[0m {text}")
    
    def _load_audio(self, audio_path: str):
        """Load audio file"""
        import librosa
        audio, sr = librosa.load(audio_path, sr=None)
        return audio, sr
    
    def start_voice_conversation(self):
        """Start a voice-to-voice conversation"""
        print("\n" + "=" * 70)
        print("  [35mVoice Conversation Mode[0m")
        print("=" * 70)
        print("  [33mInstructions:[0m")
        print("  - Speak naturally after the 'Listening...' prompt")
        print("  - Wait for Aira to respond")
        print("  - Say 'goodbye' or 'quit' to end the conversation")
        print("  - Press Ctrl+C to stop at any time")
        print("=" * 70 + "\n")
        
        # Initial greeting
        initial_text = "Hello! I'm Aira, your voice AI assistant. How can I help you today?"
        self._speak(initial_text, emotion='happy')
        
        try:
            while True:
                # Record user input
                audio_path = self._record_audio(duration=10.0)
                
                if audio_path is None:
                    print("No audio recorded. Try again.")
                    continue
                
                # Detect emotion
                emotion = self._detect_emotion(audio_path)
                print(f"\n[33mDetected emotion: {emotion['emotion']}[0m")
                
                # Transcribe
                transcription = self._transcribe(audio_path)
                user_text = transcription.get('text', '').strip()
                
                # Cleanup audio file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                
                if not user_text:
                    print("I didn't catch that. Could you please repeat?")
                    self._speak("I didn't catch that. Could you please repeat?", emotion='neutral')
                    continue
                
                print(f"\n[36mYou:[0m {user_text}")
                
                # Check for exit commands
                if user_text.lower() in ['goodbye', 'bye', 'quit', 'exit', 'stop']:
                    response = "Goodbye! It was nice talking with you. Have a great day!"
                    print(f"\n[34mAira:[0m {response}")
                    self._speak(response, emotion='happy')
                    break
                
                # Generate response
                response = self._generate_response(user_text, emotion)
                
                # Speak response
                self._speak(response, emotion=emotion.get('emotion', 'neutral'))
                
        except KeyboardInterrupt:
            print("\n\nConversation ended by user.")
            self._speak("Conversation ended. Goodbye!", emotion='neutral')
        except Exception as e:
            print(f"\nError during conversation: {e}")
            self._speak("I'm sorry, I encountered an error. Let's try again.", emotion='sad')
    
    def start_text_conversation(self):
        """Start a text-based conversation with voice output"""
        print("\n" + "=" * 70)
        print("  [35mText-to-Voice Conversation Mode[0m")
        print("=" * 70)
        print("  [33mInstructions:[0m")
        print("  - Type your messages and press Enter")
        print("  - Aira will respond with voice")
        print("  - Type 'quit' or 'exit' to end the conversation")
        print("=" * 70 + "\n")
        
        # Initial greeting
        initial_text = "Hello! I'm Aira, your AI assistant. Type your message and I'll respond with voice."
        self._speak(initial_text, emotion='happy')
        
        try:
            while True:
                user_text = input("\n[36mYou:[0m ").strip()
                
                if not user_text:
                    continue
                
                # Check for exit commands
                if user_text.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                    response = "Goodbye! Have a great day!"
                    print(f"\n[34mAira:[0m {response}")
                    self._speak(response, emotion='happy')
                    break
                
                # Generate response
                emotion = {'emotion': 'neutral'}
                response = self._generate_response(user_text, emotion)
                
                # Speak response
                self._speak(response, emotion='neutral')
                
        except KeyboardInterrupt:
            print("\n\nConversation ended by user.")
            self._speak("Conversation ended. Goodbye!", emotion='neutral')
    
    def start_voice_command_mode(self):
        """Start voice command mode (single commands)"""
        print("\n" + "=" * 70)
        print("  [35mVoice Command Mode[0m")
        print("=" * 70)
        print("  [33mInstructions:[0m")
        print("  - Speak a command after the 'Listening...' prompt")
        print("  - Aira will execute the command and respond")
        print("  - Say 'exit' or press Ctrl+C to quit")
        print("=" * 70 + "\n")
        
        self._speak("Voice command mode activated. What would you like me to do?", emotion='neutral')
        
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
                    self._speak("I didn't hear a command.", emotion='neutral')
                    continue
                
                print(f"\n[36mCommand:[0m {user_text}")
                
                # Check for exit
                if user_text in ['exit', 'quit', 'stop', 'goodbye', 'bye']:
                    self._speak("Exiting voice command mode. Goodbye!", emotion='happy')
                    break
                
                # Process command
                response = self._process_command(user_text)
                self._speak(response, emotion='neutral')
                
        except KeyboardInterrupt:
            print("\n\nVoice command mode ended.")
            self._speak("Voice command mode ended.", emotion='neutral')
    
    def _process_command(self, command: str) -> str:
        """Process voice command"""
        command_lower = command.lower()
        
        # Greetings
        if any(word in command_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you?"
        
        # Time
        if any(word in command_lower for word in ['time', 'what time']):
            from datetime import datetime
            now = datetime.now()
            return f"The current time is {now.strftime('%H:%M')}."
        
        # Date
        if any(word in command_lower for word in ['date', 'what day', 'today']):
            from datetime import datetime
            now = datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."
        
        # Jokes
        if any(word in command_lower for word in ['joke', 'funny']):
            import random
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "What do you call a fake noodle? An impasta!",
                "How do you organize a space party? You planet!",
                "Why can't you give Elsa a balloon? Because she will let it go!"
            ]
            return random.choice(jokes)
        
        # Help
        if any(word in command_lower for word in ['help', 'what can you do']):
            return "I can tell you the time, date, jokes, and help with various tasks. Just ask!"
        
        # Default
        return f"I heard you say: '{command}'. I'm not sure how to help with that yet."
    
    def shutdown(self):
        """Shutdown Voice AI"""
        print("\nShutting down Voice AI...")
        
        if self.agi_agent:
            try:
                self.agi_agent.shutdown()
            except:
                pass
        
        print("Voice AI shutdown complete.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Voice-to-Voice AI Assistant")
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
    parser.add_argument("--stt-model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--tts-rate", type=int, default=170,
                        help="TTS speech rate (words per minute)")
    parser.add_argument("--tts-volume", type=float, default=0.9,
                        help="TTS volume (0.0 to 1.0)")
    parser.add_argument("--female", action="store_true", default=True,
                        help="Use young female voice for TTS")
    
    args = parser.parse_args()
    
    # Create Voice AI instance
    voice_ai = VoiceAI(
        use_agi=not args.no_agi,
        use_stt=not args.no_stt,
        use_tts=not args.no_tts,
        use_emotion=not args.no_emotion,
        stt_model=args.stt_model,
        tts_rate=args.tts_rate,
        tts_volume=args.tts_volume,
        young_female=args.female
    )
    
    try:
        # Start appropriate mode
        if args.mode == "voice":
            voice_ai.start_voice_conversation()
        elif args.mode == "text":
            voice_ai.start_text_conversation()
        elif args.mode == "command":
            voice_ai.start_voice_command_mode()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        voice_ai.shutdown()


if __name__ == "__main__":
    main()

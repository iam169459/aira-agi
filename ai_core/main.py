"""
Main AI Integration - Brings all modules together
Voice-to-voice conversation AI with emotion awareness, screen perception, 
desktop automation, and self-healing capabilities
"""
import os
import sys
import time
import signal
import argparse
from typing import Optional, Dict, Any

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.emotion import EmotionFeatureExtractor, MoodState
from modules.stt import STTModule
from modules.tts import TTSModule
from modules.screen_perception import ScreenCapture, ContentAnalyzer, PerceptionState
from modules.desktop_automation import DesktopController, SafeModeGuard
from modules.self_healing import SelfHealingCode, ErrorInfo
from ai_core.agent import AiraAgent
import torch


class EmotionAwareAI:
    """Main AI class integrating all capabilities"""
    
    def __init__(self, 
                 model_name="base",
                 emotion_model=None,
                 use_gpu=False):
        
        self.healer = SelfHealingCode()
        self.running = False
        
        # Initialize core components
        print("Initializing AI components...")
        
        # STT
        self.stt = STTModule(model_size=model_name)
        
        # TTS
        self.tts = TTSModule(rate=170, volume=0.9, use_offline=False, young_female=True)
        
        # Emotion perception
        if emotion_model is None:
            self.emotion_extractor = EmotionFeatureExtractor(n_mfcc=40)
        else:
            self.emotion_extractor = emotion_model
        
        self.mood_state = MoodState()
        
        # Screen perception
        self.screen_capture = ScreenCapture()
        self.content_analyzer = ContentAnalyzer()
        self.perception_state = PerceptionState()
        
        # Desktop automation
        self.desktop = DesktopController(speed=1.0)
        self.safe_mode = SafeModeGuard()
        
        # AI model
        self.vocab_size = 50257  # GPT-2 vocab size example
        self.cfg = {
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "block_size": 512,
            "dropout": 0.1
        }
        self._model = None
        self._brain = None
        
        print("AI system ready!")
    
    @property
    def model(self):
        """Lazily allocate the transformer on first use"""
        if self._model is None:
            self._model = self._init_ai_model()
        return self._model
    
    def _init_ai_model(self):
        """Initialize the core AI model"""
        from ai_core.model import EmotionAwareGPT
        model = EmotionAwareGPT(self.vocab_size, self.cfg)
        return model
    
    def process_speech_input(self, audio_path: str) -> Dict[str, Any]:
        """Process speech input and extract text + emotion"""
        # Transcribe
        stt_result = self.stt.transcribe(audio_path)
        text = stt_result.get('text', '')
        
        # Detect emotion from audio
        try:
            emotion_logits, mood = self.emotion_extractor.extract_mfcc(
                torch.randn(16000)  # placeholder
            )
            # In a real implementation, use actual audio features here
            detected_emotion = self.mood_state.update(emotion_logits, mood)
            confidence = 0.5
        except Exception as e:
            detected_emotion = 'neutral'
            confidence = 0.5
            print(f"Emotion detection error: {e}")
        
        return {
            'text': text,
            'emotion': detected_emotion,
            'mood': self.mood_state.get_mood_description(),
            'valence': self.mood_state.valence,
            'arousal': self.mood_state.arousal,
            'confidence': confidence
        }
    
    @property
    def brain(self) -> AiraAgent:
        """Lazily created conversation brain (model-backed when a
        checkpoint exists, template voice otherwise)"""
        if self._brain is None:
            self._brain = AiraAgent()
        return self._brain
    
    def generate_response(self, user_text: str, mood: str) -> str:
        """Generate an AI response through the shared conversation brain.

        Model-backed generation when a ``train.py`` checkpoint exists,
        mood-aware template voice otherwise — one source of truth with
        ``ai_core.agent.AiraAgent``.
        """
        return self.brain.think(user_text, emotion=mood)
    
    def perceive_screen(self) -> Dict[str, Any]:
        """Capture and analyze screen content"""
        # Capture screen
        image = self.screen_capture.capture()
        
        if image is not None:
            # Analyze content
            analysis = self.content_analyzer.analyze(image)
            
            # Extract text
            extracted_text = self.content_analyzer.extract_text(image)
            
            # Update perception state
            self.perception_state.update(image, analysis)
            
            return {
                'image': image,
                'analysis': analysis,
                'extracted_text': extracted_text,
                'mood': self.mood_state.get_mood_description()
            }
        
        return {'analysis': {}, 'extracted_text': ''}
    
    def execute_safe_action(self, action: str, **kwargs) -> bool:
        """Execute action with safe-mode guardrails"""
        if not self.safe_mode.can_execute(action, **kwargs):
            print("Safe-mode: Action blocked for safety")
            return False
        
        # Execute the action
        try:
            if action == "click":
                self.desktop.click(**kwargs)
            elif action == "type":
                self.desktop.type(**kwargs)
            elif action == "press":
                self.desktop.press(**kwargs)
            elif action == "drag":
                self.desktop.drag_to(**kwargs)
            elif action == "scroll":
                self.desktop.scroll(**kwargs)
            elif action == "hotkey":
                self.desktop.press_hotkey(**kwargs)
            else:
                print(f"Unknown action: {action}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Action execution error: {e}")
            # Attempt self-healing
            self.healer.attempt_fix(e, f"action:{action}")
            return False
    
    def converse(self, user_text: str) -> str:
        """Main conversation loop - generate response"""
        # Detect mood from text content (simplified)
        mood = self.mood_state.get_mood_description()
        
        # Generate response
        response = self.generate_response(user_text, mood)
        
        # Update mood based on conversation
        # In a real system, this would use the emotion extractor
        
        return response
    
    def start_voice_loop(self):
        """Start the voice conversation loop"""
        print("Starting voice conversation loop...")
        print("Say 'exit' or 'quit' to stop")
        
        self.running = True
        
        while self.running:
            try:
                # In a real implementation, this would capture from microphone
                # For now, simulate with text input
                try:
                    user_input = input("\nYou: ")
                except EOFError:
                    print("\nAI: End of input. Goodbye!")
                    self.running = False
                    break
                
                if user_input.lower() in ['exit', 'quit', 'goodbye']:
                    print("AI: Goodbye! Have a great day!")
                    self.running = False
                    break
                
                # Generate response
                response = self.converse(user_input)
                print(f"AI: {response}")
                
                try:
                    # Speak response
                    self.tts.speak_immediate(response)
                except Exception:
                    pass
                
            except KeyboardInterrupt:
                print("\nAI: Interrupted. Goodbye!")
                self.running = False
            except Exception as e:
                print(f"Error: {e}")
                # Attempt self-healing
                try:
                    self.healer.attempt_fix(e, "voice_loop")
                except Exception:
                    pass
                time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description='Emotion-Aware AI Assistant')
    parser.add_argument('--model', type=str, default='base', 
                        help='Whisper model size (tiny, base, small, medium, large)')
    parser.add_argument('--voice', action='store_true',
                        help='Enable voice input/output')
    parser.add_argument('--text', action='store_true',
                        help='Text-only mode')
    parser.add_argument('--emotion', action='store_true',
                        help='Enable emotion detection')
    
    args = parser.parse_args()
    
    # Create AI instance
    ai = EmotionAwareAI(
        model_name=args.model,
        use_gpu=False  # Set to True if GPU available
    )
    
    if args.text:
        # Text-only mode
        print("Text mode active. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                response = ai.converse(user_input)
                print(f"AI: {response}")
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                break
    else:
        # Voice mode
        try:
            ai.start_voice_loop()
        except EOFError:
            pass
        except Exception as e:
            print(f"Error: {e}")
            pass


if __name__ == "__main__":
    main()
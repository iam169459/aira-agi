"""
Main AI Integration - Brings all modules together
Voice-to-voice conversation AI with emotion awareness, screen perception, 
desktop automation, and self-healing capabilities
"""
import sys
import time
import signal
import argparse
from typing import Optional, Dict, Any

# Add modules to path
sys.path.insert(0, '/workspaces/tiny-from-scratch')

from modules.emotion import EmotionFeatureExtractor, MoodState
from modules.stt import STTModule
from modules.tts import TTSModule
from modules.screen_perception import ScreenCapture, ContentAnalyzer, PerceptionState
from modules.desktop_automation import DesktopController, SafeModeGuard
from modules.self_healing import SelfHealingCode, ErrorInfo
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
        self.model = self._init_ai_model()
        
        print("AI system ready!")
    
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
            # In real implementation, use actual audio
            emotion_name = 'neutral'
            confidence = 0.5
        except Exception as e:
            emotion_name = 'neutral'
            confidence = 0.5
            print(f"Emotion detection error: {e}")
        
        # Update mood state
        if 'mood' in dir():
            detected_emotion = self.mood_state.update(
                emotion_logits, mood
            )
        else:
            detected_emotion = emotion_name
        
        return {
            'text': text,
            'emotion': detected_emotion,
            'mood': self.mood_state.get_mood_description(),
            'valence': self.mood_state.valence,
            'arousal': self.mood_state.arousal,
            'confidence': confidence
        }
    
    def generate_response(self, user_text: str, mood: str) -> str:
        """Generate AI response based on input and mood"""
        # In a real implementation, this would use the EmotionAwareGPT model
        # For now, use template-based response with mood adaptation
        
        mood_responses = {
            "stressed": "I can hear you're feeling stressed. Take a deep breath. How can I help?",
            "angry": "I sense your frustration. Let me help you with that. What seems to be the problem?",
            "excited": "Wow, you sound excited! Tell me more about what's got you so pumped!",
            "happy": "That's great to hear! What's making you so happy?",
            "calm": "I can sense you're in a calm state. How can I assist you today?",
            "neutral": "I'm listening. What's on your mind?",
            "tired": "You sound tired. Maybe you should rest. Is there anything I can help with before you go?",
            "sad": "I'm sorry you're feeling down. I'm here if you want to talk or need help with anything."
        }
        
        # Get base response for mood
        base_response = mood_responses.get(mood, mood_responses["neutral"])
        
        # Adapt based on user text content
        response = base_response
        
        # Simple keyword-based adaptation
        user_lower = user_text.lower()
        if any(word in user_lower for word in ['help', 'assist', 'support']):
            if mood in ["stressed", "angry", "sad"]:
                response = "I'm here to help you. What specifically do you need assistance with?"
            else:
                response = "I'd be happy to help you with that! What do you need?"
        
        if any(word in user_lower for word in ['thank', 'thanks']):
            response = "You're welcome! Is there anything else I can help you with?"
        
        return response
    
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
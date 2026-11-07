#!/usr/bin/env python3
"""
Demo: Aira's Girl Persona

This script demonstrates Aira's new girl persona with:
- Sweet, intelligent, playful personality
- Girl-like speech patterns
- Emotional responses
- Natural conversations
"""

import sys
import os
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Girl Persona
try:
    from aira_persona import GirlPersona, Emotion, PersonaConfig
    PERSONA_AVAILABLE = True
except ImportError as e:
    print(f"Error: Girl Persona not available: {e}")
    PERSONA_AVAILABLE = False
    sys.exit(1)


def print_header(title: str):
    """Print a colorful header"""
    print("\n" + "=" * 70)
    print(f"  [35m[1m{title}[0m")
    print("=" * 70 + "\n")


def print_conversation(user: str, aira: str):
    """Print conversation in a nice format"""
    print(f"\n[36mYou:[0m {user}")
    print(f"\n[35m[1mAira:[0m {aira}")
    time.sleep(1.5)


def demo_basic_conversation():
    """Demonstrate basic conversation"""
    print_header("Basic Conversation Demo")
    
    persona = GirlPersona()
    
    conversations = [
        ("Hi there!", None),
        ("What's your name?", None),
        ("How old are you?", None),
        ("How are you today?", None),
        ("Thank you!", None),
    ]
    
    for user_text, emotion in conversations:
        response = persona.generate_response(user_text)
        print_conversation(user_text, response)


def demo_emotional_conversation():
    """Demonstrate emotional responses"""
    print_header("Emotional Responses Demo")
    
    persona = GirlPersona()
    
    conversations = [
        ("I'm so happy today!", Emotion.HAPPY),
        ("I just got a promotion!", Emotion.EXCITED),
        ("I had a really bad day", Emotion.SAD),
        ("My phone broke", Emotion.SAD),
        ("That's so annoying!", Emotion.ANGRY),
        ("Tell me a joke", Emotion.PLAYFUL),
        ("You're so cute", None),  # Will trigger blush
        ("What's your favorite color?", Emotion.CURIOUS),
    ]
    
    for user_text, emotion in conversations:
        if emotion:
            persona.current_emotion = emotion
        response = persona.generate_response(user_text)
        print_conversation(user_text, response)


def demo_girl_specific_patterns():
    """Demonstrate girl-specific speech patterns"""
    print_header("Girl-Specific Patterns Demo")
    
    persona = GirlPersona()
    
    conversations = [
        ("Bye!", None),
        ("See you later", None),
        ("Thanks a lot", None),
        ("You're beautiful", None),
        ("I'm sorry", None),
        ("My name is John", None),
        ("I'm 25 years old", None),
        ("What can you do?", None),
    ]
    
    for user_text, emotion in conversations:
        response = persona.generate_response(user_text)
        print_conversation(user_text, response)


def demo_emotion_detection():
    """Demonstrate emotion detection from text"""
    print_header("Emotion Detection Demo")
    
    persona = GirlPersona()
    
    test_texts = [
        "I'm so excited about my birthday party!",
        "I'm really sad because I lost my dog",
        "I'm so angry at my boss!",
        "That's the funniest thing I've ever heard!",
        "I'm a little nervous about the test tomorrow",
        "I feel so embarrassed about what I said",
        "I'm really proud of my accomplishment",
        "I'm just feeling okay today",
    ]
    
    for text in test_texts:
        emotion = persona.detect_emotion_from_text(text)
        response = persona.generate_response(text, emotion)
        print_conversation(text, response)
        print(f"  [33m[Detected emotion: {emotion.value}][0m")


def demo_persona_info():
    """Show persona information"""
    print_header("Persona Information")
    
    persona = GirlPersona()
    info = persona.get_persona_info()
    
    print(f"\n[33mName:[0m {info['name']}")
    print(f"\n[33mAge:[0m {info['age']}")
    print(f"\n[33mGender:[0m {info['gender']}")
    print(f"\n[33mPersonality:[0m {info['personality']}")
    print(f"\n[33mCurrent Emotion:[0m {info['current_emotion']}")
    print(f"\n[33mEmotion Intensity:[0m {info['emotion_intensity']:.1f}")
    
    print("\n")


def demo_custom_persona():
    """Demonstrate custom persona configuration"""
    print_header("Custom Persona Demo")
    
    # Create a custom persona
    config = PersonaConfig(
        name="Aira",
        age=17,
        personality="sweet, intelligent, playful, empathetic",
        
        # Voice characteristics
        voice_rate=180,
        voice_volume=0.95,
        
        # Behavior characteristics
        politeness_level=0.95,
        playfulness_level=0.8,
        empathy_level=0.85,
        shyness_level=0.2,
        
        # Response characteristics
        use_emojis=True,
        use_exclamations=True,
        use_contractions=True,
        use_slang=True,
        use_filler_words=True,
    )
    
    persona = GirlPersona(config)
    
    print("Custom Persona Configuration:")
    print(f"  Age: {config.age}")
    print(f"  Voice Rate: {config.voice_rate} WPM")
    print(f"  Politeness: {config.politeness_level:.0%}")
    print(f"  Playfulness: {config.playfulness_level:.0%}")
    print(f"  Empathy: {config.empathy_level:.0%}")
    print(f"  Shyness: {config.shyness_level:.0%}")
    
    print("\nConversation with custom persona:")
    conversations = [
        "Hi!",
        "What's up?",
        "You're amazing!",
        "Tell me something fun",
    ]
    
    for user_text in conversations:
        response = persona.generate_response(user_text)
        print_conversation(user_text, response)


def demo_voice_properties():
    """Demonstrate voice properties for different emotions"""
    print_header("Voice Properties Demo")
    
    persona = GirlPersona()
    
    emotions = [
        Emotion.HAPPY,
        Emotion.EXCITED,
        Emotion.SAD,
        Emotion.ANGRY,
        Emotion.PLAYFUL,
        Emotion.SHY,
        Emotion.CURIOUS,
        Emotion.SURPRISED,
    ]
    
    print("\nVoice properties for different emotions:")
    print("-" * 70)
    
    for emotion in emotions:
        persona.current_emotion = emotion
        properties = persona.get_voice_properties()
        
        print(f"\n[33m{emotion.value.upper()}[0m:")
        print(f"  Rate: {properties['rate']} WPM")
        print(f"  Volume: {properties['volume']:.2f}")
        print(f"  Pitch: {properties['pitch']}")


def demo_all_features():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("  [35m[1mAIRA'S GIRL PERSONA - FULL DEMO[0m")
    print("=" * 70)
    
    demo_persona_info()
    demo_basic_conversation()
    demo_girl_specific_patterns()
    demo_emotional_conversation()
    demo_emotion_detection()
    demo_voice_properties()
    demo_custom_persona()
    
    print("\n" + "=" * 70)
    print("  [35m[1mDEMO COMPLETE![0m")
    print("=" * 70)
    print("\nNow try the real thing:")
    print("  [36mpython voice_ai_girl.py --mode voice[0m")
    print("\nOr read the full guide:")
    print("  [36mcat GIRL_PERSONA_GUIDE.md[0m")
    print("=" * 70 + "\n")


def interactive_demo():
    """Interactive demo with user input"""
    print_header("Interactive Demo")
    
    persona = GirlPersona()
    
    print("Type your messages to chat with Aira. Type 'quit' to exit.")
    print("-" * 70)
    
    while True:
        try:
            user_text = input("\n[36mYou:[0m ").strip()
            
            if not user_text:
                continue
            
            if user_text.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                response = persona.generate_response(user_text)
                print_conversation(user_text, response)
                break
            
            # Detect emotion from text
            emotion = persona.detect_emotion_from_text(user_text)
            
            # Generate response
            response = persona.generate_response(user_text, emotion)
            
            print_conversation(user_text, response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Aira's Girl Persona Demo")
    parser.add_argument("--basic", action="store_true", help="Run basic conversation demo")
    parser.add_argument("--emotional", action="store_true", help="Run emotional responses demo")
    parser.add_argument("--patterns", action="store_true", help="Run girl-specific patterns demo")
    parser.add_argument("--emotion-detection", action="store_true", help="Run emotion detection demo")
    parser.add_argument("--voice", action="store_true", help="Run voice properties demo")
    parser.add_argument("--custom", action="store_true", help="Run custom persona demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat demo")
    
    args = parser.parse_args()
    
    if not any([args.basic, args.emotional, args.patterns, args.emotion_detection, 
                 args.voice, args.custom, args.all, args.interactive]):
        # Default: show persona info and basic demo
        demo_persona_info()
        demo_basic_conversation()
        print("\nFor more demos, run:")
        print("  python demo_girl_persona.py --all")
        print("\nOr for interactive chat:")
        print("  python demo_girl_persona.py --interactive")
    else:
        if args.all:
            demo_all_features()
        else:
            if args.basic:
                demo_basic_conversation()
            if args.emotional:
                demo_emotional_conversation()
            if args.patterns:
                demo_girl_specific_patterns()
            if args.emotion_detection:
                demo_emotion_detection()
            if args.voice:
                demo_voice_properties()
            if args.custom:
                demo_custom_persona()
            if args.interactive:
                interactive_demo()


if __name__ == "__main__":
    import argparse
    import random
    
    # Seed for reproducible demos
    random.seed(42)
    
    main()

"""
Tiny From Scratch - AI Project with Multi-Modal Capabilities

A comprehensive AI project featuring:
- Voice-to-voice conversation with STT and TTS
- Emotion-aware responses
- Screen perception and analysis
- Desktop automation
- Self-healing code
- Safe-mode guardrails

Based on the "Tiny From Scratch" educational GPT project,
extended with modern AI capabilities.
"""

__version__ = "2.0.0"

import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def check_requirements():
    """Check if required packages are installed"""
    print("Checking requirements...")
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        import whisper
        print("Whisper (STT) available")
    except ImportError:
        print("Installing Whisper...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], check=True)
        import whisper
    
    try:
        import pyttsx3
        print("pyttsx3 (TTS) available")
    except ImportError:
        print("Installing pyttsx3...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pyttsx3"], check=True)
    
    try:
        import cv2
        print("OpenCV (screen perception) available")
    except ImportError:
        print("Installing OpenCV...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python"], check=True)
    
    print("Requirements check complete!\n")

def main():
    """Main entry point"""
    print("=" * 60)
    print("Tiny From Scratch - Multi-Modal AI Assistant")
    print("Version:" + __version__)
    print("=" * 60)
    
    # Check and install requirements
    check_requirements()
    
    # Import and run main AI
    from ai_core.main import main as ai_main
    ai_main()

if __name__ == "__main__":
    main()
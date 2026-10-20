"""
Speech-to-Text Module
Uses OpenAI Whisper for transcription
"""
import torch
import whisper
import tempfile
import os
from typing import Optional


class STTModule:
    """Speech-to-Text using Whisper"""
    
    def __init__(self, model_size="base", device=None):
        """
        Initialize Whisper model
        model_size: "tiny", "base", "small", "medium", "large"
        device: "cpu", "cuda", or None (auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper {model_size} on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("STT model loaded successfully!")
    
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio file
        Returns dict with: text, language, duration, segments
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        result = self.model.transcribe(audio_path)
        return result
    
    def transcribe_from_array(self, audio_array: torch.Tensor, sample_rate: int = 16000) -> dict:
        """
        Transcribe from audio tensor
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            # Save as WAV
            import soundfile as sf
            sf.write(tmp_path, audio_array.numpy(), sample_rate)
        
        try:
            return self.transcribe(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def list_available_models(self):
        """List available Whisper model sizes"""
        return ["tiny", "base", "small", "medium", "large"]


class EmotionFromAudio:
    """Detect emotion from audio features"""
    
    def __init__(self):
        self.feature_extractor = None  # EmotionFeatureExtractor would go here
    
    def detect_emotion(self, audio_path: str):
        """Detect emotion from audio file"""
        # Placeholder - integrate with emotion module later
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'valence': 0.0,
            'arousal': 0.5
        }
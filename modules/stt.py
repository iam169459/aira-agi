"""
Speech-to-Text Module
Uses OpenAI Whisper for transcription
"""
import tempfile
import os
from typing import Optional

try:
    import numpy
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import torch
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    torch = None
    whisper = None
    WHISPER_AVAILABLE = False

try:
    from .natural_speech import SpeechRecognizer as _NaturalRecognizer
    NATURAL_STT_AVAILABLE = True
except ImportError:
    _NaturalRecognizer = None
    NATURAL_STT_AVAILABLE = False

# Fix SSL certificate verification for model downloads on macOS (self-signed chain issue)
if WHISPER_AVAILABLE:
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


class STTModule:
    """Speech-to-Text using Whisper"""
    
    def __init__(self, model_size="base", device=None, vocabulary=None):
        """
        Initialize speech recognition.
        model_size: "tiny", "base", "small", "medium", "large" (Whisper)
        device: "cpu", "cuda", or None (auto-detect)
        vocabulary: optional word list; when Whisper is unavailable, the
            built-in from-scratch recognizer (natural_speech) is used with
            this vocabulary (default: its built-in command vocabulary).
        """
        self.vocabulary = vocabulary
        if not WHISPER_AVAILABLE:
            if not NATURAL_STT_AVAILABLE:
                raise ImportError(
                    "Neither whisper nor the natural_speech fallback is "
                    "available; install openai-whisper to use STTModule.")
            print("Whisper unavailable; using built-in from-scratch STT "
                  "(natural_speech, vocabulary-based). Enroll your voice "
                  "with .natural_recognizer.enroll() for best accuracy.")
            self.model = None
            self.natural_recognizer = _NaturalRecognizer(vocabulary=vocabulary)
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper {model_size} on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        self.natural_recognizer = None
        print("STT model loaded successfully!")
    
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio file
        Returns dict with: text, language, duration, segments
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.model is None:
            return self.natural_recognizer.transcribe(audio_path)
        
        result = self.model.transcribe(audio_path)
        return result
    
    def transcribe_from_array(self, audio_array, sample_rate: int = 16000) -> dict:
        """
        Transcribe from audio tensor (torch tensor) or numpy array
        """
        if self.model is None:
            array = audio_array
            if hasattr(array, "numpy"):
                array = array.numpy()
            samples, sr = array, sample_rate
            if _NUMPY_AVAILABLE:
                import numpy as _np
                if array.ndim == 2:
                    samples = array.mean(axis=0)
            return self.natural_recognizer.transcribe(samples, sample_rate=sr)
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
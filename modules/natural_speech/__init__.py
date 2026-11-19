"""natural_speech: a 100% from-scratch TTS + STT engine.

No external speech APIs, models, or pretrained data:
- TTS:  text normalization -> rule-based grapheme-to-phoneme -> prosody model
        -> source-filter formant synthesis -> WAV.
- STT:  energy VAD -> MFCC features (own FFT/mel/DCT) -> banded DTW word
        matching against templates rendered by this engine's own TTS, with
        optional user voice enrollment.

Only numpy + Python stdlib are used.
"""
from .dtw import dtw_cost
from .features import MFCCExtractor
from .stt import SpeechRecognizer, DEFAULT_VOCABULARY
from .synth import FormantSynthesizer
from .tts import TextToSpeech, load_wav, resample, save_wav

__all__ = [
    "TextToSpeech", "SpeechRecognizer", "FormantSynthesizer",
    "MFCCExtractor", "dtw_cost", "load_wav", "save_wav", "resample",
    "DEFAULT_VOCABULARY",
]

__version__ = "1.2.0"

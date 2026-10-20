import numpy as np
import librosa
import sounddevice as sd
import queue
import threading

class EmotionPerception:
    """
    Handles microphone input and extracts emotional features from audio.
    """
    def __init__(self, sample_rate=22050, duration=3):
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

    def record_chunk(self):
        """Records a short chunk of audio for analysis."""
        recording = sd.rec(int(self.duration * self.sample_rate), 
                           samplerate=self.sample_rate, channels=1)
        sd.wait()
        return recording.flatten()

    def extract_features(self, audio_data):
        """Extracts MFCCs, Chroma, Mel, and Contrast features."""
        stft = np.abs(librosa.stft(audio_data))
        
        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=40).T, axis=0)
        
        # Chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=self.sample_rate).T, axis=0)
        
        # Mel
        mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate).T, axis=0)
        
        # Contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=self.sample_rate).T, axis=0)
        
        # Tonnetz
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=self.sample_rate).T, axis=0)
        
        return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

    def classify_emotion(self, audio_data):
        """
        Classifies the emotion based on extracted features.
        In a real scenario, this would use a pre-trained model.
        For this prototype, we'll use a placeholder heuristic.
        """
        # Placeholder: In a real implementation, load a sklearn/tf model here
        # features = self.extract_features(audio_data)
        
        # Energy and pitch can give us some basic clues
        rms = np.sqrt(np.mean(audio_data**2))
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Heuristic for demo purposes
        if rms > 0.05:
            if pitch > 300: return "excited"
            if pitch < 150: return "angry"
            return "stressed"
        else:
            if pitch < 100: return "tired"
            if pitch > 200: return "happy"
            return "neutral"

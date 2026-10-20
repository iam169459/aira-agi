"""
Emotion Perception Module
Detects emotion from audio features
"""
import torch
import torch.nn as nn
import librosa
import numpy as np


class EmotionFeatureExtractor(nn.Module):
    """Extracts emotion-relevant features from audio"""
    
    def __init__(self, n_mfcc=40, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # CNN for feature extraction from spectrograms
        self.conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Emotion classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 8)  # 8 emotion categories
        )
        
        # Mood regression head (valence, arousal, dominance)
        self.mood_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
    
    def extract_mfcc(self, audio):
        """Extract MFCC features from audio"""
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=16000, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc  # shape: (n_mfcc, time_steps)
    
    def forward(self, audio_tensor):
        """
        audio_tensor: (batch, audio_length) or (audio_length,)
        Returns: emotion_logits (batch, 8), mood (batch, 3)
        """
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Extract MFCC features
        mfcc = self.extract_mfcc(audio_tensor.squeeze().numpy())
        mfcc_tensor = torch.from_numpy(mfcc).float().unsqueeze(0)  # (1, n_mfcc, time)
        
        # Pass through CNN
        x = F.relu(self.conv1(mfcc_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global average pooling
        x = x.mean(dim=-1)  # (1, channels, 1) -> (1, channels)
        
        # Get emotion logits and mood
        emotion_logits = self.classifier(x.squeeze(0))  # (8,)
        mood = self.mood_regressor(x.squeeze(0))  # (3,)
        
        return emotion_logits, mood


class MoodState:
    """Persistent mood engine with valence, arousal, dominance dimensions"""
    
    EMOTION_CATEGORIES = ['stressed', 'angry', 'excited', 'happy', 'calm', 'neutral', 'tired', 'sad']
    
    def __init__(self, initial_valence=0.5, initial_arousal=0.5, initial_dominance=0.5, decay_rate=0.95):
        self.valence = initial_valence  # -1 to 1 (negative to positive)
        self.arousal = initial_arousal  # 0 to 1 (calm to excited)
        self.dominance = initial_dominance  # 0 to 1 (powerless to powerful)
        self.decay_rate = decay_rate
        self.emotion_history = []
    
    def update(self, emotion_logits, mood):
        """Update mood state from emotion detection"""
        # Get dominant emotion
        emo_idx = torch.argmax(emotion_logits).item()
        emotion_name = self.EMOTION_CATEGORIES[emo_idx]
        
        # Update mood from regression output
        self.valence = float(torch.sigmoid(mood[0]) * 2 - 1)  # -1 to 1
        self.arousal = float(torch.sigmoid(mood[1]))  # 0 to 1
        self.dominance = float(torch.sigmoid(mood[2]))  # 0 to 1
        
        # Apply decay toward neutral
        self.valence = self.valence * self.decay_rate + 0 * (1 - self.decay_rate)
        self.arousal = self.arousal * self.decay_rate + 0.5 * (1 - self.decay_rate)
        self.dominance = self.dominance * self.decay_rate + 0.5 * (1 - self.decay_rate)
        
        # Store history
        self.emotion_history.append({
            'emotion': emotion_name,
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance
        })
        
        # Keep only recent history
        if len(self.emotion_history) > 50:
            self.emotion_history = self.emotion_history[-50:]
        
        return emotion_name
    
    def get_mood_description(self):
        """Get human-readable mood description"""
        emotion_idx = int((self.arousal * 7 + self.valence * 3 + self.dominance * 2) % 8)
        return self.EMOTION_CATEGORIES[emotion_idx]
    
    def to_dict(self):
        """Serialize mood state"""
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'emotion_history': self.emotion_history
        }
    
    @classmethod
    def from_dict(cls, data):
        """Deserialize mood state"""
        mood = cls(
            initial_valence=data.get('valence', 0.5),
            initial_arousal=data.get('arousal', 0.5),
            initial_dominance=data.get('dominance', 0.5)
        )
        mood.emotion_history = data.get('emotion_history', [])
        return mood
"""
Audio preprocessing utilities for mental health audio dataset
"""

import librosa
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


class AudioPreprocessor:
    """Handles audio loading, normalization, and mel spectrogram conversion"""
    
    def __init__(self, target_sr=16000, n_fft=1024, hop_length=512, n_mels=80, max_length_sec=10):
        """
        Args:
            target_sr: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            max_length_sec: Maximum audio length in seconds
        """
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_length = int(max_length_sec * target_sr)
    
    def load_and_normalize(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and convert to target sample rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Normalized audio array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        # Pad or trim to max length
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.max_length]
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def audio_to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to log mel spectrogram
        
        Args:
            audio: Audio array
            
        Returns:
            Log mel spectrogram (n_mels, time)
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1]
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)
        log_mel_spec = 2 * log_mel_spec - 1
        
        return log_mel_spec
    
    def mel_to_audio(
        self,
        mel_spec: np.ndarray,
        n_iter: int = 32,
        output_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert mel spectrogram back to audio using Griffin-Lim
        
        Args:
            mel_spec: Log mel spectrogram (n_mels, time)
            n_iter: Number of Griffin-Lim iterations
            output_length: Optional target waveform length in samples.
            
        Returns:
            Reconstructed audio
        """
        # Denormalize from [-1, 1] to [min_db, max_db]
        mel_spec = (mel_spec + 1) / 2  # [0, 1]
        # Assume original range was [-80, 0] dB
        mel_spec = mel_spec * 80 - 80
        
        # Convert from dB to power
        mel_spec_power = librosa.db_to_power(mel_spec)
        
        # Inverse mel to linear spectrogram
        spec = librosa.feature.inverse.mel_to_stft(
            mel_spec_power, 
            sr=self.target_sr, 
            n_fft=self.n_fft,
            fmax=8000
        )
        
        # Griffin-Lim algorithm
        audio = librosa.griffinlim(
            spec,
            n_iter=n_iter,
            hop_length=self.hop_length,
            length=output_length
        )
        
        return audio
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio to file"""
        sf.write(output_path, audio, self.target_sr)
    
    def process_file(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio, mel_spectrogram)
        """
        audio = self.load_and_normalize(audio_path)
        mel_spec = self.audio_to_mel_spectrogram(audio)
        return audio, mel_spec


def compute_audio_stats(audio_dir: str, preprocessor: AudioPreprocessor) -> dict:
    """
    Compute dataset statistics for normalization
    
    Args:
        audio_dir: Directory containing audio files
        preprocessor: AudioPreprocessor instance
        
    Returns:
        Dictionary with mean and std statistics
    """
    audio_files = list(Path(audio_dir).rglob("*.wav")) + list(Path(audio_dir).rglob("*.mp3"))
    
    all_mels = []
    
    for audio_file in audio_files[:100]:  # Sample first 100 files
        try:
            _, mel = preprocessor.process_file(str(audio_file))
            all_mels.append(mel)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    if not all_mels:
        return {"mean": 0.0, "std": 1.0}
    
    all_mels = np.concatenate([m.flatten() for m in all_mels])
    
    return {
        "mean": float(np.mean(all_mels)),
        "std": float(np.std(all_mels))
    }

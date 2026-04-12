"""Utilities for frame-level audio to semantic-feature preprocessing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler


class SemanticAudioPreprocessor:
    """Preprocess raw waveforms and frame-level semantic feature CSVs."""

    def __init__(
        self,
        target_sr: int = 16000,
        frame_size_ms: float = 25.0,
        hop_size_ms: float = 10.0,
    ):
        self.target_sr = target_sr
        self.frame_size_ms = frame_size_ms
        self.hop_size_ms = hop_size_ms
        self.frame_size_samples = int(round(target_sr * (frame_size_ms / 1000.0)))
        self.hop_size_samples = int(round(target_sr * (hop_size_ms / 1000.0)))

        if self.frame_size_samples <= 0 or self.hop_size_samples <= 0:
            raise ValueError("Frame and hop sizes must be positive")
        if self.frame_size_samples < self.hop_size_samples:
            raise ValueError("Frame size must be >= hop size")

    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load mono waveform at target sample rate and normalize to [-1, 1]."""
        audio, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
        if audio.size == 0:
            return np.zeros(self.frame_size_samples, dtype=np.float32)

        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 0:
            audio = audio / max_abs
        return audio.astype(np.float32)

    def frame_audio(self, audio: np.ndarray) -> np.ndarray:
        """Split waveform into overlapping frames of shape (num_frames, frame_size_samples)."""
        if audio.size < self.frame_size_samples:
            pad_width = self.frame_size_samples - audio.size
            audio = np.pad(audio, (0, pad_width), mode="constant")

        num_frames = 1 + (audio.size - self.frame_size_samples) // self.hop_size_samples
        if num_frames <= 0:
            return np.zeros((0, self.frame_size_samples), dtype=np.float32)

        strides = (audio.strides[0] * self.hop_size_samples, audio.strides[0])
        shape = (num_frames, self.frame_size_samples)
        framed = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        return np.array(framed, dtype=np.float32, copy=True)

    def load_feature_csv(
        self,
        csv_path: str,
        drop_columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Load frame-level feature CSV and keep only numeric feature columns."""
        drop_set = {"name", "frameTime"}
        if drop_columns:
            drop_set.update(drop_columns)

        rows: List[List[float]] = []
        csv_file = Path(csv_path)
        with csv_file.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            if reader.fieldnames is None:
                return np.zeros((0, 0), dtype=np.float32)

            feature_columns = [c for c in reader.fieldnames if c not in drop_set]
            for row in reader:
                values: List[float] = []
                valid_row = True
                for col in feature_columns:
                    cell = row.get(col, "")
                    try:
                        values.append(float(cell))
                    except (TypeError, ValueError):
                        valid_row = False
                        break
                if valid_row:
                    rows.append(values)

        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def align_frames(audio_frames: np.ndarray, feature_frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Trim audio/feature frames to the same number of frames."""
        aligned_len = min(len(audio_frames), len(feature_frames))
        if aligned_len <= 0:
            return (
                np.zeros((0, audio_frames.shape[1] if audio_frames.ndim == 2 else 0), dtype=np.float32),
                np.zeros((0, feature_frames.shape[1] if feature_frames.ndim == 2 else 0), dtype=np.float32),
            )
        return audio_frames[:aligned_len], feature_frames[:aligned_len]


class FeatureNormalizer:
    """Standard scaling for frame-level semantic features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, features: np.ndarray) -> None:
        if features.size == 0:
            raise ValueError("Cannot fit normalizer on empty features")
        self.scaler.fit(features)
        self.is_fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transform")
        if features.size == 0:
            return features.astype(np.float32)
        return self.scaler.transform(features).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

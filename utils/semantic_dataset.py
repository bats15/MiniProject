"""Dataset for frame-level waveform to semantic-feature mapping on DAIC-WOZ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from utils.semantic_preprocessing import FeatureNormalizer, SemanticAudioPreprocessor


@dataclass
class ParticipantSample:
    participant_id: str
    audio_path: Path
    feature_path: Path


class DAICSemanticMappingDataset(Dataset):
    """Frame-level dataset where X is waveform frames and y is semantic feature frames."""

    def __init__(
        self,
        data_dir: str,
        audio_subdir: str = "audio",
        feature_subdir: str = "features",
        feature_type: str = "egemaps",
        target_sr: int = 16000,
        frame_size_ms: float = 25.0,
        hop_size_ms: float = 10.0,
        normalize_features: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / audio_subdir
        self.feature_dir = self.data_dir / feature_subdir
        self.feature_type = feature_type.lower()
        if self.feature_type not in {"egemaps", "mfcc"}:
            raise ValueError("feature_type must be 'egemaps' or 'mfcc'")

        self.preprocessor = SemanticAudioPreprocessor(
            target_sr=target_sr,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
        )
        self.normalize_features = normalize_features
        self.normalizer = FeatureNormalizer() if normalize_features else None
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if not self.audio_dir.exists():
            raise ValueError(f"Audio directory not found: {self.audio_dir}")
        if not self.feature_dir.exists():
            raise ValueError(f"Feature directory not found: {self.feature_dir}")

        self.participants = self._discover_participants()
        if not self.participants:
            raise ValueError(f"No valid DAIC participants found in {self.data_dir}")

        self.X, self.y, self.index_meta = self._build_frame_dataset()

    def _discover_participants(self) -> List[ParticipantSample]:
        participants: List[ParticipantSample] = []

        for audio_path in sorted(self.audio_dir.glob("*.wav")):
            participant_id = audio_path.stem.split("_", maxsplit=1)[0]

            if self.feature_type == "egemaps":
                candidate_patterns = [
                    f"{participant_id}_OpenSMILE2.3.0_egemaps.csv",
                    f"{participant_id}_OpenSMILE2.3.0_eGeMAPS.csv",
                ]
            else:
                candidate_patterns = [
                    f"{participant_id}_OpenSMILE2.3.0_mfcc.csv",
                    f"{participant_id}_OpenSMILE2.3.0_MFCC.csv",
                ]

            feature_path = None
            for name in candidate_patterns:
                path = self.feature_dir / name
                if path.exists():
                    feature_path = path
                    break

            if feature_path is None:
                continue

            participants.append(
                ParticipantSample(
                    participant_id=participant_id,
                    audio_path=audio_path,
                    feature_path=feature_path,
                )
            )

        return participants

    def _participant_cache_path(self, participant_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        cache_root = self.cache_dir / "semantic_mapping"
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"{participant_id}_{self.feature_type}_{self.preprocessor.target_sr}.npz"

    def _load_participant_arrays(self, sample: ParticipantSample) -> Tuple[np.ndarray, np.ndarray]:
        cache_path = self._participant_cache_path(sample.participant_id)
        if cache_path is not None and cache_path.exists():
            cache = np.load(cache_path)
            return cache["X"].astype(np.float32), cache["y"].astype(np.float32)

        audio = self.preprocessor.load_audio(str(sample.audio_path))
        audio_frames = self.preprocessor.frame_audio(audio)
        feature_frames = self.preprocessor.load_feature_csv(str(sample.feature_path))
        audio_frames, feature_frames = self.preprocessor.align_frames(audio_frames, feature_frames)

        if cache_path is not None:
            np.savez_compressed(cache_path, X=audio_frames, y=feature_frames)

        return audio_frames, feature_frames

    def _build_frame_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
        x_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []
        index_meta: List[Dict[str, int]] = []

        for sample in tqdm(self.participants, desc="Building semantic mapping dataset"):
            x_part, y_part = self._load_participant_arrays(sample)
            if x_part.size == 0 or y_part.size == 0:
                continue

            x_chunks.append(x_part)
            y_chunks.append(y_part)
            index_meta.extend(
                {"participant": int(sample.participant_id), "frame": i}
                for i in range(len(x_part))
            )

        if not x_chunks or not y_chunks:
            raise ValueError("No aligned audio/feature frames available")

        x_total = np.concatenate(x_chunks, axis=0).astype(np.float32)
        y_total = np.concatenate(y_chunks, axis=0).astype(np.float32)

        if self.normalize_features and self.normalizer is not None:
            y_total = self.normalizer.fit_transform(y_total)

        return x_total, y_total, index_meta

    @property
    def frame_samples(self) -> int:
        return int(self.X.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.y.shape[1])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "waveform_frame": torch.from_numpy(self.X[idx]),
            "semantic_target": torch.from_numpy(self.y[idx]),
        }


def create_semantic_mapping_dataloaders(
    data_dir: str,
    audio_subdir: str,
    feature_subdir: str,
    feature_type: str,
    batch_size: int,
    num_workers: int,
    target_sr: int = 16000,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0,
    normalize_features: bool = True,
    cache_dir: Optional[str] = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DAICSemanticMappingDataset, DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders from combined frame-level dataset."""
    if abs((train_split + val_split + test_split) - 1.0) > 1e-6:
        raise ValueError("train_split + val_split + test_split must equal 1.0")

    dataset = DAICSemanticMappingDataset(
        data_dir=data_dir,
        audio_subdir=audio_subdir,
        feature_subdir=feature_subdir,
        feature_type=feature_type,
        target_sr=target_sr,
        frame_size_ms=frame_size_ms,
        hop_size_ms=hop_size_ms,
        normalize_features=normalize_features,
        cache_dir=cache_dir,
    )

    total_len = len(dataset)
    train_len = int(total_len * train_split)
    val_len = int(total_len * val_split)
    test_len = total_len - train_len - val_len

    if train_len <= 0 or val_len <= 0 or test_len <= 0:
        raise ValueError("Dataset split produced empty partition. Add more data or adjust split ratios.")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset,
        lengths=[train_len, val_len, test_len],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, train_loader, val_loader, test_loader

"""Models for waveform-frame to semantic-feature regression."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BottleneckHead(nn.Module):
    """Optional compression bottleneck with additive Gaussian channel noise."""

    def __init__(self, in_dim: int, out_dim: int, latent_dim: int = 0, noise_std: float = 0.0):
        super().__init__()
        self.use_bottleneck = latent_dim > 0
        self.noise_std = float(noise_std)

        if self.use_bottleneck:
            self.to_latent = nn.Linear(in_dim, latent_dim)
            self.to_output = nn.Linear(latent_dim, out_dim)
            self.latent_dim = latent_dim
        else:
            self.to_output = nn.Linear(in_dim, out_dim)
            self.latent_dim = in_dim

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_bottleneck:
            z = self.to_latent(hidden)
        else:
            z = hidden

        if self.training and self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std

        y_hat = self.to_output(z)
        return y_hat, z


class WaveformToSemanticCNN(nn.Module):
    """1D CNN baseline for frame-wise semantic feature prediction."""

    def __init__(
        self,
        frame_samples: int,
        feature_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 0,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.frame_samples = frame_samples
        self.feature_dim = feature_dim

        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = _BottleneckHead(
            in_dim=hidden_dim,
            out_dim=feature_dim,
            latent_dim=latent_dim,
            noise_std=noise_std,
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        x = x.unsqueeze(1)
        hidden = self.backbone(x).squeeze(-1)
        y_hat, z = self.head(hidden)
        if return_latent:
            return y_hat, z
        return y_hat


class WaveformToSemanticCNNLSTM(nn.Module):
    """CNN feature extractor with LSTM temporal modeling inside each frame."""

    def __init__(
        self,
        frame_samples: int,
        feature_dim: int,
        conv_dim: int = 64,
        lstm_hidden_dim: int = 128,
        latent_dim: int = 0,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, conv_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = _BottleneckHead(
            in_dim=lstm_hidden_dim,
            out_dim=feature_dim,
            latent_dim=latent_dim,
            noise_std=noise_std,
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        x = x.unsqueeze(1)
        conv_out = self.conv(x)
        sequence = conv_out.transpose(1, 2)
        _, (h_n, _) = self.lstm(sequence)
        hidden = h_n[-1]
        y_hat, z = self.head(hidden)
        if return_latent:
            return y_hat, z
        return y_hat


class WaveformToSemanticTransformer(nn.Module):
    """Lightweight Transformer encoder for frame-wise waveform mapping."""

    def __init__(
        self,
        frame_samples: int,
        feature_dim: int,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        latent_dim: int = 0,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Conv1d(1, model_dim, kernel_size=4, stride=4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.head = _BottleneckHead(
            in_dim=model_dim,
            out_dim=feature_dim,
            latent_dim=latent_dim,
            noise_std=noise_std,
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        x = x.unsqueeze(1)
        tokens = self.proj(x).transpose(1, 2)
        encoded = self.encoder(tokens)
        hidden = self.norm(encoded.mean(dim=1))
        y_hat, z = self.head(hidden)
        if return_latent:
            return y_hat, z
        return y_hat


class SpectrogramSemanticCommNet(nn.Module):
    """Spectrogram-domain semantic communication pipeline.

    Semantic Extraction -> Compression -> Channel -> Interpolation -> Denoising -> Reconstruction
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        feature_dim: int,
        encoder_channels: int = 64,
        latent_dim: int = 64,
        noise_std: float = 0.0,
        interpolation_scale: float = 1.0,
        denoiser_channels: int = 64,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError("SpectrogramSemanticCommNet expects input_shape=(n_mels, patch_frames)")

        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.noise_std = float(noise_std)
        self.interpolation_scale = float(interpolation_scale)

        self.semantic_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, encoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(),
        )

        self.compressor = nn.Conv1d(encoder_channels, latent_dim, kernel_size=1)

        self.denoiser = nn.Sequential(
            nn.Conv1d(latent_dim, denoiser_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(denoiser_channels, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.reconstructor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, feature_dim),
        )

    def _channel(self, z_seq: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            z_seq = z_seq + torch.randn_like(z_seq) * self.noise_std
        return z_seq

    def _interpolate(self, z_seq: torch.Tensor) -> torch.Tensor:
        if self.interpolation_scale <= 0:
            raise ValueError("interpolation_scale must be positive")
        if abs(self.interpolation_scale - 1.0) < 1e-6:
            return z_seq
        return F.interpolate(
            z_seq,
            scale_factor=self.interpolation_scale,
            mode="linear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError("Expected spectrogram input with shape [B, n_mels, patch_frames]")

        x = x.unsqueeze(1)
        feat_map = self.semantic_extractor(x)

        temporal_tokens = feat_map.mean(dim=2)

        z_seq = self.compressor(temporal_tokens)
        z_seq = self._channel(z_seq)
        z_seq = self._interpolate(z_seq)
        z_seq = self.denoiser(z_seq)

        y_hat = self.reconstructor(z_seq)
        z = z_seq.mean(dim=-1)
        if return_latent:
            return y_hat, z
        return y_hat


def build_semantic_mapper(
    architecture: str,
    frame_samples: Optional[int],
    feature_dim: int,
    input_shape: Optional[Tuple[int, ...]] = None,
    latent_dim: int = 0,
    noise_std: float = 0.0,
    interpolation_scale: float = 1.0,
    denoiser_channels: int = 64,
) -> nn.Module:
    """Factory for semantic mapper architectures."""
    arch = architecture.lower()

    common_kwargs: Dict[str, float] = {
        "latent_dim": latent_dim,
        "noise_std": noise_std,
    }

    if arch == "cnn":
        if frame_samples is None:
            raise ValueError("frame_samples is required for cnn architecture")
        return WaveformToSemanticCNN(frame_samples, feature_dim, **common_kwargs)
    if arch == "cnn_lstm":
        if frame_samples is None:
            raise ValueError("frame_samples is required for cnn_lstm architecture")
        return WaveformToSemanticCNNLSTM(frame_samples, feature_dim, **common_kwargs)
    if arch == "transformer":
        if frame_samples is None:
            raise ValueError("frame_samples is required for transformer architecture")
        return WaveformToSemanticTransformer(frame_samples, feature_dim, **common_kwargs)
    if arch == "spectrogram_semcom":
        if input_shape is None:
            raise ValueError("input_shape is required for spectrogram_semcom architecture")
        return SpectrogramSemanticCommNet(
            input_shape=input_shape,
            feature_dim=feature_dim,
            latent_dim=latent_dim if latent_dim > 0 else 64,
            noise_std=noise_std,
            interpolation_scale=interpolation_scale,
            denoiser_channels=denoiser_channels,
        )

    raise ValueError(f"Unsupported architecture: {architecture}")

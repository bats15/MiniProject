"""Models for waveform-frame to semantic-feature regression."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


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


def build_semantic_mapper(
    architecture: str,
    frame_samples: int,
    feature_dim: int,
    latent_dim: int = 0,
    noise_std: float = 0.0,
) -> nn.Module:
    """Factory for semantic mapper architectures."""
    arch = architecture.lower()

    common_kwargs: Dict[str, float] = {
        "latent_dim": latent_dim,
        "noise_std": noise_std,
    }

    if arch == "cnn":
        return WaveformToSemanticCNN(frame_samples, feature_dim, **common_kwargs)
    if arch == "cnn_lstm":
        return WaveformToSemanticCNNLSTM(frame_samples, feature_dim, **common_kwargs)
    if arch == "transformer":
        return WaveformToSemanticTransformer(frame_samples, feature_dim, **common_kwargs)

    raise ValueError(f"Unsupported architecture: {architecture}")

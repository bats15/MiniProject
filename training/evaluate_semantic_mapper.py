"""Evaluate waveform-to-semantic mapper checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.semantic_mapper import build_semantic_mapper
from utils.semantic_dataset import create_semantic_mapping_dataloaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/semantic_mapping.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, _, _, test_loader = create_semantic_mapping_dataloaders(
        data_dir=config["paths"]["data_dir"],
        audio_subdir=config["paths"]["audio_subdir"],
        feature_subdir=config["paths"]["feature_subdir"],
        feature_type=config["semantic_mapping"]["feature_type"],
        input_representation=config["semantic_mapping"].get("input_representation", "spectrogram"),
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        target_sr=config["semantic_mapping"]["target_sr"],
        frame_size_ms=config["semantic_mapping"]["frame_size_ms"],
        hop_size_ms=config["semantic_mapping"]["hop_size_ms"],
        n_mels=config["semantic_mapping"].get("n_mels", 64),
        spectrogram_patch_frames=config["semantic_mapping"].get("spectrogram_patch_frames", 5),
        spectrogram_n_fft=config["semantic_mapping"].get("spectrogram_n_fft"),
        normalize_features=config["semantic_mapping"]["normalize_features"],
        cache_dir=config["semantic_mapping"].get("cache_dir"),
        train_split=config["semantic_mapping"]["train_split"],
        val_split=config["semantic_mapping"]["val_split"],
        test_split=config["semantic_mapping"]["test_split"],
        seed=config["semantic_mapping"].get("seed", 42),
    )

    model = build_semantic_mapper(
        architecture=config["semantic_mapping"]["architecture"],
        frame_samples=dataset.frame_samples,
        feature_dim=dataset.feature_dim,
        input_shape=dataset.input_shape,
        latent_dim=config["semantic_mapping"].get("latent_dim", 0),
        noise_std=0.0,
        interpolation_scale=config["semantic_mapping"].get("interpolation_scale", 1.0),
        denoiser_channels=config["semantic_mapping"].get("denoiser_channels", 64),
    ).to(device)

    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            x = batch.get("model_input", batch.get("waveform_frame")).to(device)
            y = batch["semantic_target"].to(device)
            y_hat = model(x)

            mse = F.mse_loss(y_hat, y)
            mae = F.l1_loss(y_hat, y)

            batch_size = x.size(0)
            total_mse += float(mse.item()) * batch_size
            total_mae += float(mae.item()) * batch_size
            total += batch_size

    print(f"Test MSE: {total_mse / max(total, 1):.6f}")
    print(f"Test MAE: {total_mae / max(total, 1):.6f}")


if __name__ == "__main__":
    main()

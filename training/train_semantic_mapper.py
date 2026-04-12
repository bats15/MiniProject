"""Train waveform-to-semantic feature mapper for DAIC-WOZ."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.semantic_mapper import build_semantic_mapper
from utils.semantic_dataset import create_semantic_mapping_dataloaders


class SemanticMapperTrainer:
    """Trainer for frame-level waveform to semantic feature reconstruction."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        train_cfg = self.config["semantic_mapping"]
        configured_device = self.config.get("device", "cuda")
        if torch.cuda.is_available():
            if isinstance(configured_device, str) and configured_device.startswith("cuda"):
                self.device = torch.device(configured_device)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.device.type == "cuda":
            dev_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(dev_idx)
            total_mem_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
            print(f"CUDA device: {gpu_name} | VRAM: {total_mem_gb:.2f} GB")

        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"]) / "semantic_mapper"
        log_dir = Path(self.config["paths"]["log_dir"]) / "semantic_mapper"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(log_dir=str(log_dir))

        self.dataset, self.train_loader, self.val_loader, self.test_loader = create_semantic_mapping_dataloaders(
            data_dir=self.config["paths"]["data_dir"],
            audio_subdir=self.config["paths"]["audio_subdir"],
            feature_subdir=self.config["paths"]["feature_subdir"],
            feature_type=train_cfg["feature_type"],
            input_representation=train_cfg.get("input_representation", "spectrogram"),
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"]["num_workers"],
            target_sr=train_cfg["target_sr"],
            frame_size_ms=train_cfg["frame_size_ms"],
            hop_size_ms=train_cfg["hop_size_ms"],
            n_mels=train_cfg.get("n_mels", 64),
            spectrogram_patch_frames=train_cfg.get("spectrogram_patch_frames", 5),
            spectrogram_n_fft=train_cfg.get("spectrogram_n_fft"),
            normalize_features=train_cfg["normalize_features"],
            cache_dir=train_cfg.get("cache_dir"),
            train_split=train_cfg["train_split"],
            val_split=train_cfg["val_split"],
            test_split=train_cfg["test_split"],
            seed=train_cfg.get("seed", 42),
        )

        self.model = build_semantic_mapper(
            architecture=train_cfg["architecture"],
            frame_samples=self.dataset.frame_samples,
            feature_dim=self.dataset.feature_dim,
            input_shape=self.dataset.input_shape,
            latent_dim=train_cfg.get("latent_dim", 0),
            noise_std=train_cfg.get("channel_noise_std", 0.0),
            interpolation_scale=train_cfg.get("interpolation_scale", 1.0),
            denoiser_channels=train_cfg.get("denoiser_channels", 64),
        ).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        self.num_epochs = self.config["training"]["num_epochs"]
        self.best_val_mse = float("inf")
        self._shape_logged = False

        print(f"Device: {self.device}")
        print(f"Dataset size: {len(self.dataset)} frames")
        print(f"Input shape: {self.dataset.input_shape}")
        print(f"Target feature dim: {self.dataset.feature_dim}")

    def _run_epoch(self, dataloader, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        sum_mse = 0.0
        sum_mae = 0.0
        total = 0

        for batch in tqdm(dataloader, desc="train" if train else "val"):
            x = batch.get("model_input", batch.get("waveform_frame")).to(self.device)
            y = batch["semantic_target"].to(self.device)

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                y_hat = self.model(x)
                if not self._shape_logged:
                    print(
                        f"Data flow | X: {tuple(x.shape)} -> y_hat: {tuple(y_hat.shape)} -> y: {tuple(y.shape)}"
                    )
                    self._shape_logged = True
                mse = F.mse_loss(y_hat, y)
                mae = F.l1_loss(y_hat, y)
                if train:
                    mse.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip"],
                    )
                    self.optimizer.step()

            batch_size = x.size(0)
            sum_mse += float(mse.item()) * batch_size
            sum_mae += float(mae.item()) * batch_size
            total += batch_size

        return {
            "mse": sum_mse / max(total, 1),
            "mae": sum_mae / max(total, 1),
        }

    @torch.no_grad()
    def evaluate_test(self) -> Dict[str, float]:
        return self._run_epoch(self.test_loader, train=False)

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "frame_samples": self.dataset.frame_samples,
            "feature_dim": self.dataset.feature_dim,
        }

        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(payload, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(payload, best_path)

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"train_mse={train_metrics['mse']:.6f}, train_mae={train_metrics['mae']:.6f} | "
                f"val_mse={val_metrics['mse']:.6f}, val_mae={val_metrics['mae']:.6f}"
            )

            self.writer.add_scalar("train/mse", train_metrics["mse"], epoch)
            self.writer.add_scalar("train/mae", train_metrics["mae"], epoch)
            self.writer.add_scalar("val/mse", val_metrics["mse"], epoch)
            self.writer.add_scalar("val/mae", val_metrics["mae"], epoch)

            is_best = val_metrics["mse"] < self.best_val_mse
            if is_best:
                self.best_val_mse = val_metrics["mse"]

            if (epoch + 1) % self.config["training"]["checkpoint_interval"] == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best=is_best)

        test_metrics = self.evaluate_test()
        print(f"Test MSE: {test_metrics['mse']:.6f}")
        print(f"Test MAE: {test_metrics['mae']:.6f}")

        self.writer.add_hparams(
            {
                "architecture": self.config["semantic_mapping"]["architecture"],
                "feature_type": self.config["semantic_mapping"]["feature_type"],
                "latent_dim": self.config["semantic_mapping"].get("latent_dim", 0),
                "noise_std": self.config["semantic_mapping"].get("channel_noise_std", 0.0),
            },
            {
                "hparam/test_mse": test_metrics["mse"],
                "hparam/test_mae": test_metrics["mae"],
            },
        )
        self.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/semantic_mapping.yaml")
    args = parser.parse_args()

    trainer = SemanticMapperTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()

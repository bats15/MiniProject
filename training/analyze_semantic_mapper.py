"""Structured analysis for trained semantic mapper checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
import librosa
import soundfile as sf

from models.semantic_mapper import build_semantic_mapper
from utils.semantic_dataset import create_semantic_mapping_dataloaders

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _forward_with_latent_noise(model: torch.nn.Module, x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Inject Gaussian noise at latent stage for semcom model without changing config."""
    if noise_std <= 0:
        return model(x)

    if hasattr(model, "semantic_extractor") and hasattr(model, "compressor"):
        x2 = x.unsqueeze(1)
        feat_map = model.semantic_extractor(x2)
        temporal_tokens = feat_map.mean(dim=2)
        z_seq = model.compressor(temporal_tokens)
        z_seq = z_seq + torch.randn_like(z_seq) * float(noise_std)
        z_seq = model._interpolate(z_seq)
        z_seq = model.denoiser(z_seq)
        y_hat = model.reconstructor(z_seq)
        return y_hat

    return model(x)


def _evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    noise_std: float = 0.0,
) -> Tuple[float, float]:
    model.eval()
    total = 0
    sum_mse = 0.0
    sum_mae = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch.get("model_input", batch.get("waveform_frame")).to(device)
            y = batch["semantic_target"].to(device)
            y_hat = _forward_with_latent_noise(model, x, noise_std=noise_std)

            mse = torch.mean((y_hat - y) ** 2, dim=1)
            mae = torch.mean(torch.abs(y_hat - y), dim=1)
            n = x.size(0)
            sum_mse += float(mse.sum().item())
            sum_mae += float(mae.sum().item())
            total += n
    return sum_mse / max(total, 1), sum_mae / max(total, 1)


def _pearson_per_feature(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    corr = np.zeros(pred.shape[1], dtype=np.float32)
    for i in range(pred.shape[1]):
        p = pred[:, i]
        t = target[:, i]
        p_std = float(np.std(p))
        t_std = float(np.std(t))
        if p_std < 1e-8 or t_std < 1e-8:
            corr[i] = 0.0
        else:
            corr[i] = float(np.corrcoef(p, t)[0, 1])
    return corr


def _save_noise_csv(noise_results: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    rows = ["noise_std,mse,mae"]
    for k in sorted(noise_results.keys(), key=lambda x: float(x)):
        rows.append(f"{k},{noise_results[k]['mse']:.8f},{noise_results[k]['mae']:.8f}")
    (out_dir / "noise_robustness.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _save_best_worst_csv(best_samples: List[Dict[str, float]], worst_samples: List[Dict[str, float]], out_dir: Path) -> None:
    header = "group,subset_pos,base_idx,participant,frame,mse"
    rows = [header]
    for rec in best_samples:
        rows.append(
            f"best,{rec['subset_pos']},{rec['base_idx']},{rec['participant']},{rec['frame']},{rec['mse']:.8f}"
        )
    for rec in worst_samples:
        rows.append(
            f"worst,{rec['subset_pos']},{rec['base_idx']},{rec['participant']},{rec['frame']},{rec['mse']:.8f}"
        )
    (out_dir / "best_worst_samples.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _save_plots(
    pred_all: np.ndarray,
    tgt_all: np.ndarray,
    feature_mae: np.ndarray,
    best_features: List[int],
    worst_features: List[int],
    noise_results: Dict[str, Dict[str, float]],
    out_dir: Path,
) -> None:
    if plt is None:
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = np.array([float(k) for k in sorted(noise_results.keys(), key=lambda x: float(x))], dtype=np.float32)
    noise_mse = np.array([noise_results[f"{x:.2f}"]["mse"] for x in noise_levels], dtype=np.float32)
    plt.figure(figsize=(7, 4))
    plt.plot(noise_levels, noise_mse, marker="o")
    plt.title("Noise Robustness Curve")
    plt.xlabel("Latent Noise Std")
    plt.ylabel("Test MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "noise_robustness_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(feature_mae)), feature_mae)
    plt.title("Mean Absolute Error by Feature Index")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean MAE")
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_mae_bar.png", dpi=160)
    plt.close()

    trace_len = min(400, pred_all.shape[0])
    trace_idx = best_features[:2] + worst_features[:2]
    plt.figure(figsize=(10, 6))
    for i, fi in enumerate(trace_idx):
        ax = plt.subplot(len(trace_idx), 1, i + 1)
        ax.plot(tgt_all[:trace_len, fi], label="target", linewidth=1.5)
        ax.plot(pred_all[:trace_len, fi], label="pred", linewidth=1.0)
        ax.set_ylabel(f"f{fi}")
        if i == 0:
            ax.legend(loc="upper right")
    plt.xlabel("Frame")
    plt.suptitle("Predicted vs Target Feature Traces")
    plt.tight_layout()
    plt.savefig(plot_dir / "pred_vs_target_traces.png", dpi=160)
    plt.close()


def _resolve_audio_path(dataset, participant: int) -> Path:
    part_str = str(participant)
    for sample in dataset.participants:
        if sample.participant_id == part_str:
            return sample.audio_path
    raise ValueError(f"Audio path not found for participant {participant}")


def _extract_reference_clip(
    audio_path: Path,
    frame_index: int,
    target_sr: int,
    frame_size_samples: int,
    hop_size_samples: int,
) -> np.ndarray:
    audio, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    start = frame_index * hop_size_samples
    end = start + frame_size_samples
    if start >= len(audio):
        return np.zeros(frame_size_samples, dtype=np.float32)
    clip = audio[start:end]
    if len(clip) < frame_size_samples:
        clip = np.pad(clip, (0, frame_size_samples - len(clip)), mode="constant")
    return clip.astype(np.float32)


def _reconstruct_from_mel_patch(
    mel_patch_db: np.ndarray,
    target_sr: int,
    n_fft: int,
    hop_size_samples: int,
    frame_size_samples: int,
) -> np.ndarray:
    mel_power = librosa.db_to_power(mel_patch_db)
    wav = librosa.feature.inverse.mel_to_audio(
        M=mel_power,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_size_samples,
        win_length=frame_size_samples,
        center=False,
    )
    if wav.size == 0:
        return np.zeros(frame_size_samples, dtype=np.float32)
    peak = float(np.max(np.abs(wav)))
    if peak > 1e-8:
        wav = wav / peak
    return wav.astype(np.float32)


def _save_audio_previews(
    dataset,
    best_samples: List[Dict[str, float]],
    worst_samples: List[Dict[str, float]],
    target_sr: int,
    n_fft: int,
    out_dir: Path,
) -> None:
    audio_dir = out_dir / "audio_previews"
    audio_dir.mkdir(parents=True, exist_ok=True)

    frame_size_samples = dataset.preprocessor.frame_size_samples
    hop_size_samples = dataset.preprocessor.hop_size_samples

    for group_name, records in (("best", best_samples), ("worst", worst_samples)):
        for rank, rec in enumerate(records, start=1):
            base_idx = int(rec["base_idx"])
            participant = int(rec["participant"])
            frame = int(rec["frame"])

            mel_patch = dataset.X[base_idx]
            recon_wav = _reconstruct_from_mel_patch(
                mel_patch_db=mel_patch,
                target_sr=target_sr,
                n_fft=n_fft,
                hop_size_samples=hop_size_samples,
                frame_size_samples=frame_size_samples,
            )

            audio_path = _resolve_audio_path(dataset, participant)
            ref_wav = _extract_reference_clip(
                audio_path=audio_path,
                frame_index=frame,
                target_sr=target_sr,
                frame_size_samples=frame_size_samples,
                hop_size_samples=hop_size_samples,
            )

            prefix = f"{group_name}_{rank}_p{participant}_f{frame}"
            sf.write(audio_dir / f"{prefix}_reconstructed_from_mel.wav", recon_wav, target_sr)
            sf.write(audio_dir / f"{prefix}_reference_clip.wav", ref_wav, target_sr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/semantic_mapping.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/semantic_mapper/best.pt")
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/semantic_performance_report.json",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="analysis/artifacts",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, _, val_loader, test_loader = create_semantic_mapping_dataloaders(
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

    val_mse, val_mae = _evaluate_loader(model, val_loader, device, noise_std=0.0)
    test_mse, test_mae = _evaluate_loader(model, test_loader, device, noise_std=0.0)

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    noise_results = {}
    for ns in noise_levels:
        nmse, nmae = _evaluate_loader(model, test_loader, device, noise_std=ns)
        noise_results[f"{ns:.2f}"] = {"mse": nmse, "mae": nmae}

    test_subset = test_loader.dataset
    subset_indices: List[int] = list(getattr(test_subset, "indices", range(len(test_subset))))

    preds: List[np.ndarray] = []
    tgts: List[np.ndarray] = []
    sample_records: List[Dict[str, float]] = []
    offset = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch.get("model_input", batch.get("waveform_frame")).to(device)
            y = batch["semantic_target"].to(device)
            y_hat = model(x)

            per_sample_mse = torch.mean((y_hat - y) ** 2, dim=1).detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            yhat_np = y_hat.detach().cpu().numpy()

            preds.append(yhat_np)
            tgts.append(y_np)

            for i, mse in enumerate(per_sample_mse):
                global_pos = offset + i
                base_idx = int(subset_indices[global_pos])
                meta = dataset.index_meta[base_idx]
                sample_records.append(
                    {
                        "subset_pos": int(global_pos),
                        "base_idx": base_idx,
                        "participant": int(meta["participant"]),
                        "frame": int(meta["frame"]),
                        "mse": float(mse),
                    }
                )
            offset += int(x.size(0))

    pred_all = np.concatenate(preds, axis=0)
    tgt_all = np.concatenate(tgts, axis=0)

    feature_mae = np.mean(np.abs(pred_all - tgt_all), axis=0)
    feature_corr = _pearson_per_feature(pred_all, tgt_all)

    ranked_mae = np.argsort(feature_mae)
    best_features = ranked_mae[:5].tolist()
    worst_features = ranked_mae[-5:][::-1].tolist()

    sample_records_sorted = sorted(sample_records, key=lambda x: x["mse"])
    best_samples = sample_records_sorted[:5]
    worst_samples = sample_records_sorted[-5:][::-1]

    # Temporal consistency by participant: compare frame-to-frame deltas in prediction vs target.
    by_participant: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {}
    for rec_idx, rec in enumerate(sample_records):
        p = int(rec["participant"])
        fr = int(rec["frame"])
        by_participant.setdefault(p, []).append((fr, pred_all[rec_idx], tgt_all[rec_idx]))

    temporal_delta_gap: List[float] = []
    for _, seq in by_participant.items():
        seq_sorted = sorted(seq, key=lambda t: t[0])
        if len(seq_sorted) < 2:
            continue
        pred_seq = np.stack([x[1] for x in seq_sorted], axis=0)
        tgt_seq = np.stack([x[2] for x in seq_sorted], axis=0)
        pred_delta = np.mean(np.abs(np.diff(pred_seq, axis=0)))
        tgt_delta = np.mean(np.abs(np.diff(tgt_seq, axis=0)))
        temporal_delta_gap.append(float(pred_delta - tgt_delta))

    report = {
        "config": {
            "architecture": config["semantic_mapping"]["architecture"],
            "input_representation": config["semantic_mapping"].get("input_representation", "spectrogram"),
            "feature_type": config["semantic_mapping"]["feature_type"],
        },
        "core_metrics": {
            "val": {"mse": val_mse, "mae": val_mae},
            "test": {"mse": test_mse, "mae": test_mae},
        },
        "noise_robustness_test": noise_results,
        "feature_analysis": {
            "mean_feature_mae": feature_mae.tolist(),
            "mean_feature_corr": feature_corr.tolist(),
            "best_feature_indices_by_mae": best_features,
            "worst_feature_indices_by_mae": worst_features,
            "avg_feature_corr": float(np.mean(feature_corr)),
        },
        "sample_analysis": {
            "best_samples": best_samples,
            "worst_samples": worst_samples,
        },
        "temporal_analysis": {
            "mean_delta_gap_pred_minus_target": float(np.mean(temporal_delta_gap))
            if temporal_delta_gap
            else 0.0,
            "std_delta_gap_pred_minus_target": float(np.std(temporal_delta_gap))
            if temporal_delta_gap
            else 0.0,
        },
        "semantic_quality": {
            "avg_feature_correlation": float(np.mean(feature_corr)),
            "strong_corr_feature_count": int(np.sum(feature_corr > 0.5)),
            "weak_corr_feature_count": int(np.sum(feature_corr < 0.2)),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _save_noise_csv(report["noise_robustness_test"], artifacts_dir)
    _save_best_worst_csv(best_samples, worst_samples, artifacts_dir)
    _save_plots(
        pred_all=pred_all,
        tgt_all=tgt_all,
        feature_mae=feature_mae,
        best_features=best_features,
        worst_features=worst_features,
        noise_results=report["noise_robustness_test"],
        out_dir=artifacts_dir,
    )

    n_fft_cfg = config["semantic_mapping"].get("spectrogram_n_fft")
    if n_fft_cfg is None:
        n_fft_val = int(2 ** np.ceil(np.log2(dataset.preprocessor.frame_size_samples)))
    else:
        n_fft_val = int(n_fft_cfg)
    n_fft_val = max(n_fft_val, int(dataset.preprocessor.frame_size_samples))

    _save_audio_previews(
        dataset=dataset,
        best_samples=best_samples,
        worst_samples=worst_samples,
        target_sr=int(config["semantic_mapping"]["target_sr"]),
        n_fft=n_fft_val,
        out_dir=artifacts_dir,
    )

    print(f"Analysis written to: {out_path}")
    print(f"Artifacts written to: {artifacts_dir}")
    print(f"Validation MSE/MAE: {val_mse:.6f} / {val_mae:.6f}")
    print(f"Test MSE/MAE: {test_mse:.6f} / {test_mae:.6f}")
    print("Noise robustness (test MSE):")
    for k in sorted(report["noise_robustness_test"].keys()):
        print(f"  noise={k}: {report['noise_robustness_test'][k]['mse']:.6f}")


if __name__ == "__main__":
    main()

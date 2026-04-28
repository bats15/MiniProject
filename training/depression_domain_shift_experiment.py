"""Train depression classifier on original audio and evaluate on reconstructed audio.

Uses dataTask split CSVs (PHQ labels) and dataTask/.wav audio files.
Outputs reconstructed audio files and a structured analysis report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_split_labels(csv_path: Path) -> Dict[int, int]:
    labels: Dict[int, int] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["Participant_ID"])
            labels[pid] = int(row["PHQ8_Binary"])
    return labels


def extract_spectrogram_features(audio_path: Path, sr: int = 16000, n_mels: int = 64) -> np.ndarray:
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(n_mels * 4, dtype=np.float32)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=160)
    logmel = librosa.power_to_db(mel, ref=np.max)

    d1 = librosa.feature.delta(logmel)
    stats = [
        np.mean(logmel, axis=1),
        np.std(logmel, axis=1),
        np.mean(d1, axis=1),
        np.std(d1, axis=1),
    ]
    feat = np.concatenate(stats, axis=0).astype(np.float32)
    return feat


def reconstruct_audio_semcom_style(
    audio_path: Path,
    output_path: Path,
    sr: int = 16000,
    n_mels: int = 64,
    noise_std_db: float = 1.5,
    quant_bins: int = 64,
) -> None:
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    if y.size == 0:
        sf.write(output_path, np.zeros(sr, dtype=np.float32), sr)
        return

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=160)
    logmel = librosa.power_to_db(mel, ref=np.max)

    # Lightweight semantic-channel style corruption: compression + noise.
    q = np.round((logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-8) * (quant_bins - 1))
    logmel_q = q / max(quant_bins - 1, 1)
    logmel_q = logmel_q * (logmel.max() - logmel.min()) + logmel.min()
    logmel_q = logmel_q + np.random.normal(0.0, noise_std_db, size=logmel_q.shape).astype(np.float32)

    mel_power = librosa.db_to_power(logmel_q)
    y_rec = librosa.feature.inverse.mel_to_audio(
        M=mel_power,
        sr=sr,
        n_fft=1024,
        hop_length=160,
        win_length=1024,
    )

    if y_rec.size == 0:
        y_rec = np.zeros(sr, dtype=np.float32)

    peak = float(np.max(np.abs(y_rec)))
    if peak > 1e-8:
        y_rec = y_rec / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, y_rec.astype(np.float32), sr)


def build_dataset(
    labels: Dict[int, int],
    wav_dir: Path,
    reconstructed_dir: Path | None = None,
    use_reconstructed: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    ids: List[int] = []
    missing: List[int] = []

    for pid, label in sorted(labels.items()):
        if use_reconstructed:
            audio_path = reconstructed_dir / f"{pid}_reconstructed.wav"  # type: ignore[arg-type]
        else:
            audio_path = wav_dir / f"{pid}.wav"

        if not audio_path.exists():
            missing.append(pid)
            continue

        x_list.append(extract_spectrogram_features(audio_path))
        y_list.append(label)
        ids.append(pid)

    if not x_list:
        raise ValueError("No audio samples found for dataset build")

    return np.stack(x_list, axis=0), np.asarray(y_list, dtype=np.int64), ids, missing


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_task_dir", type=str, default="dataTask")
    parser.add_argument("--output_report", type=str, default="analysis/depression_classification_report.json")
    parser.add_argument("--reconstructed_dir", type=str, default="dataTask/reconstructed_wav")
    parser.add_argument("--noise_std_db", type=float, default=1.5)
    args = parser.parse_args()

    data_task_dir = Path(args.data_task_dir)
    wav_dir = data_task_dir / ".wav"
    train_csv = data_task_dir / "train_split_Depression_AVEC2017.csv"
    dev_csv = data_task_dir / "dev_split_Depression_AVEC2017.csv"
    reconstructed_dir = Path(args.reconstructed_dir)

    if not train_csv.exists() or not dev_csv.exists():
        raise FileNotFoundError("Missing PHQ split CSV files in dataTask directory")

    train_labels = load_split_labels(train_csv)
    dev_labels = load_split_labels(dev_csv)

    # Build reconstructed evaluation domain from original dev audio.
    reconstructed_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_map: Dict[int, str] = {}
    missing_original_for_recon: List[int] = []
    for pid in sorted(dev_labels):
        src = wav_dir / f"{pid}.wav"
        dst = reconstructed_dir / f"{pid}_reconstructed.wav"
        if not src.exists():
            missing_original_for_recon.append(pid)
            continue
        reconstruct_audio_semcom_style(
            audio_path=src,
            output_path=dst,
            noise_std_db=float(args.noise_std_db),
        )
        reconstructed_map[pid] = str(dst)

    x_train, y_train, train_ids, missing_train = build_dataset(train_labels, wav_dir)
    x_dev_orig, y_dev, dev_ids_orig, missing_dev_orig = build_dataset(dev_labels, wav_dir)
    x_dev_recon, y_dev_recon, dev_ids_recon, missing_dev_recon = build_dataset(
        dev_labels,
        wav_dir,
        reconstructed_dir=reconstructed_dir,
        use_reconstructed=True,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_dev_orig_pred = model.predict(x_dev_orig)
    y_dev_recon_pred = model.predict(x_dev_recon)

    train_metrics = evaluate(y_train, y_train_pred)
    dev_orig_metrics = evaluate(y_dev, y_dev_orig_pred)
    dev_recon_metrics = evaluate(y_dev_recon, y_dev_recon_pred)

    misclassified_orig = [
        int(pid)
        for pid, yt, yp in zip(dev_ids_orig, y_dev.tolist(), y_dev_orig_pred.tolist())
        if yt != yp
    ]
    misclassified_recon = [
        int(pid)
        for pid, yt, yp in zip(dev_ids_recon, y_dev_recon.tolist(), y_dev_recon_pred.tolist())
        if yt != yp
    ]

    report = {
        "data_task_analysis": {
            "original_audio_dir": str(wav_dir),
            "reconstructed_audio_dir": str(reconstructed_dir),
            "train_label_file": str(train_csv),
            "dev_label_file": str(dev_csv),
            "train_label_count": len(train_labels),
            "dev_label_count": len(dev_labels),
            "mapping_available": {
                "original_to_phq": True,
                "reconstructed_to_phq": True,
            },
            "missing_original_for_reconstruction": missing_original_for_recon,
            "missing_train_audio": missing_train,
            "missing_dev_original_audio": missing_dev_orig,
            "missing_dev_reconstructed_audio": missing_dev_recon,
        },
        "classification": {
            "train_original": train_metrics,
            "test_original_dev": dev_orig_metrics,
            "test_reconstructed_dev": dev_recon_metrics,
            "domain_gap_reconstructed_minus_original": {
                k: float(dev_recon_metrics[k] - dev_orig_metrics[k])
                for k in ["accuracy", "precision", "recall", "f1"]
            },
            "error_analysis": {
                "misclassified_original_dev_ids": misclassified_orig,
                "misclassified_reconstructed_dev_ids": misclassified_recon,
                "extra_errors_on_reconstructed": sorted(
                    list(set(misclassified_recon) - set(misclassified_orig))
                ),
            },
            "robustness_insight": {
                "reconstruction_harms_downstream": bool(dev_recon_metrics["f1"] < dev_orig_metrics["f1"]),
                "f1_drop": float(dev_orig_metrics["f1"] - dev_recon_metrics["f1"]),
            },
        },
    }

    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Report written to: {out_path}")
    print("Train (original):", train_metrics)
    print("Dev test (original):", dev_orig_metrics)
    print("Dev test (reconstructed):", dev_recon_metrics)
    print("Domain gap (recon - original):", report["classification"]["domain_gap_reconstructed_minus_original"])


if __name__ == "__main__":
    main()

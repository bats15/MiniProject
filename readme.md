# Audio-Based Semantic Communication Pipeline

This repository contains a cleaned, audio-only implementation for frame-level semantic communication on DAIC-WOZ style data.

The model learns:

spectrogram patch (default) or waveform frame -> semantic feature vector (eGeMAPS or MFCC)

Primary training path now enforces spectrogram-first preprocessing:

audio -> mel-spectrogram -> semantic communication mapper -> semantic feature vector

## Active Project Structure

```text
project/
├── configs/
│   └── semantic_mapping.yaml
├── data/
│   ├── audio/
│   │   └── *_AUDIO.wav
│   └── features/
│       ├── *_OpenSMILE2.3.0_egemaps.csv
│       └── *_OpenSMILE2.3.0_mfcc.csv
├── models/
│   ├── __init__.py
│   └── semantic_mapper.py
├── training/
│   ├── train_semantic_mapper.py
│   └── evaluate_semantic_mapper.py
├── utils/
│   ├── __init__.py
│   ├── semantic_dataset.py
│   └── semantic_preprocessing.py
├── main.py
├── requirements.txt
└── requirements_gpu.txt
```

## Data Placement

Place files under:

- data/audio/
- data/features/

Example pair:

- data/audio/300_AUDIO.wav
- data/features/300_OpenSMILE2.3.0_egemaps.csv

The loader matches participant ids from filenames and aligns frames by trimming to minimum length.

## Configuration

Main config file:

- configs/semantic_mapping.yaml

Key path settings:

- paths.data_dir
- paths.audio_subdir
- paths.feature_subdir

Default path resolution is fully relative for portability.

## Run

Train:

```bash
$env:PYTHONPATH='.'; & ".venv/Scripts/python.exe" main.py train --config configs/semantic_mapping.yaml
```

Evaluate:

```bash
$env:PYTHONPATH='.'; & ".venv/Scripts/python.exe" main.py eval --config configs/semantic_mapping.yaml --checkpoint checkpoints/semantic_mapper/best.pt
```

Direct scripts (optional):

```bash
$env:PYTHONPATH='.'; & ".venv/Scripts/python.exe" training/train_semantic_mapper.py --config configs/semantic_mapping.yaml
$env:PYTHONPATH='.'; & ".venv/Scripts/python.exe" training/evaluate_semantic_mapper.py --config configs/semantic_mapping.yaml --checkpoint checkpoints/semantic_mapper/best.pt
```

## Notes

- GPU is used automatically when CUDA is available.
- Training prints CUDA device and VRAM.
- Temporary data-flow shape logging is included in the trainer.
- Spectrogram conversion is the first preprocessing step for the default pipeline.
- Added semantic communication stages in the spectrogram model path: extraction -> compression -> channel noise -> interpolation -> denoising -> reconstruction.

## Training and Analysis Snapshot

- Training executed with existing config:
	- $env:PYTHONPATH='.'; & ".venv/Scripts/python.exe" main.py train --config configs/semantic_mapping.yaml
- Trained checkpoints are stored in:
	- checkpoints/semantic_mapper/
	- best model: checkpoints/semantic_mapper/best.pt
- Evaluation executed with existing config:
	- $env:PYTHONPATH='.'; & ".venv/Scripts/python.exe" main.py eval --config configs/semantic_mapping.yaml --checkpoint checkpoints/semantic_mapper/best.pt
- Structured analysis (validation/test metrics, noise robustness, temporal and feature analysis):
	- $env:PYTHONPATH='.'; & ".venv/Scripts/python.exe" training/analyze_semantic_mapper.py --config configs/semantic_mapping.yaml --checkpoint checkpoints/semantic_mapper/best.pt --output analysis/semantic_performance_report.json
	- report: analysis/semantic_performance_report.json
	- artifacts (plots, csv, and listenable reconstructed audio previews):
		- $env:PYTHONPATH='.'; & ".venv/Scripts/python.exe" training/analyze_semantic_mapper.py --config configs/semantic_mapping.yaml --checkpoint checkpoints/semantic_mapper/best.pt --output analysis/semantic_performance_report.json --artifacts_dir analysis/artifacts --preview_seconds 2.0
		- long failure-case previews (example 12s):
			- $env:PYTHONPATH='.'; & ".venv/Scripts/python.exe" training/analyze_semantic_mapper.py --config configs/semantic_mapping.yaml --checkpoint checkpoints/semantic_mapper/best.pt --output analysis/semantic_performance_report.json --artifacts_dir analysis/artifacts --preview_seconds 2.0 --long_preview_seconds 12.0
		- plots: analysis/artifacts/plots/
		- audio previews: analysis/artifacts/audio_previews/
		- long audio previews: analysis/artifacts/audio_previews_long/
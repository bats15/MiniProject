# Audio-Based Semantic Communication Pipeline

This repository contains a cleaned, audio-only implementation for frame-level semantic communication on DAIC-WOZ style data.

The model learns:

waveform frame -> semantic feature vector (eGeMAPS or MFCC)

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
python main.py train --config configs/semantic_mapping.yaml
```

Evaluate:

```bash
python main.py eval --config configs/semantic_mapping.yaml --checkpoint checkpoints/semantic_mapper/best.pt
```

Direct scripts (optional):

```bash
python training/train_semantic_mapper.py --config configs/semantic_mapping.yaml
python training/evaluate_semantic_mapper.py --config configs/semantic_mapping.yaml --checkpoint checkpoints/semantic_mapper/best.pt
```

## Notes

- GPU is used automatically when CUDA is available.
- Training prints CUDA device and VRAM.
- Temporary data-flow shape logging is included in the trainer.

## Removed from Scope

The following were removed to keep the implementation aligned with the final objective:

- visual feature assets and visual model code
- text and multimodal legacy paths
- deprecated training/inference scripts unrelated to waveform -> semantic mapping

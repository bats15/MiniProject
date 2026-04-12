# Mental Health Audio Diffusion Model

A complete pipeline for mental health analysis using diffusion models on audio data. This project implements the following components:

1. **Audio Preprocessing**: Convert audio to mel spectrograms
2. **Semantic Feature Extraction**: Extract features using Wav2Vec2, HuBERT, and OpenSMILE
3. **Feature Compression**: Encode semantic features into latent space
4. **Diffusion Model**: Generate mel spectrograms from compressed features
5. **Audio Reconstruction**: Convert mel spectrograms back to audio using Griffin-Lim or neural vocoder
6. **Mental Health Classification**: Classify audio into depression, anxiety, distress, or normal

## Project Structure

```
mental_health_audio_diffusion/
├── configs/
│   └── config.yaml              # Configuration file
├── data/                        # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── encoder.py              # Semantic feature encoder
│   ├── diffusion.py            # Diffusion model components
│   └── classifier.py           # Mental health classifier
├── utils/
│   ├── preprocessing.py        # Audio preprocessing utilities
│   ├── feature_extraction.py  # Feature extraction
│   └── dataset.py             # PyTorch dataset
├── checkpoints/                # Model checkpoints
├── outputs/                    # Generated samples
├── logs/                       # Training logs
├── train_diffusion.py         # Train diffusion model
├── train_classifier.py        # Train classifier
├── inference.py               # Inference pipeline
├── create_dummy_data.py       # Create test dataset
└── requirements.txt           # Python dependencies
```

## Installation

1. Clone the repository:
```bash
cd mental_health_audio_diffusion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install OpenSMILE:
```bash
pip install opensmile --break-system-packages
```

## Dataset Preparation

### Option 1: Use Real Dataset

Organize your audio files in the following structure:
```
data/
  train/
    depression/*.wav
    anxiety/*.wav
    distress/*.wav
    normal/*.wav
  val/
    depression/*.wav
    ...
  test/
    depression/*.wav
    ...
```

Or create annotation JSON files:
```json
[
  {
    "audio_file": "path/to/audio.wav",
    "labels": [1, 0, 0, 0],  // [depression, anxiety, distress, normal]
    "metadata": {
      "duration": 5.0,
      "sample_rate": 16000
    }
  }
]
```

### Option 2: Create Dummy Dataset for Testing

```bash
python create_dummy_data.py --output ./data --num_samples 100 --duration 5
```

This creates a synthetic dataset with 100 samples for testing the pipeline.

## Training

### 1. Train the Diffusion Model

The diffusion model learns to reconstruct audio from semantic features.

```bash
python train_diffusion.py --config configs/config.yaml
```

Options:
- `--config`: Path to configuration file
- `--resume`: Path to checkpoint to resume training

### 2. Train the Classifier

The classifier predicts mental health conditions from mel spectrograms.

```bash
python train_classifier.py --config configs/config.yaml --model basic
```

Options:
- `--config`: Path to configuration file
- `--model`: Model type (`basic`, `resnet`, `attention`)
- `--resume`: Path to checkpoint to resume training

## Inference

### Analyze a Single Audio File

```bash
python inference.py \
  --config configs/config.yaml \
  --audio path/to/audio.wav \
  --classifier checkpoints/classifier/best_model.pt \
  --output results/
```

### Batch Analysis

```bash
python inference.py \
  --config configs/config.yaml \
  --audio path/to/audio_directory/ \
  --classifier checkpoints/classifier/best_model.pt \
  --output results/ \
  --batch
```

### With Diffusion Model (Full Pipeline)

```bash
python inference.py \
  --config configs/config.yaml \
  --audio path/to/audio.wav \
  --encoder checkpoints/checkpoint_epoch10.pt \
  --diffusion checkpoints/checkpoint_epoch10.pt \
  --classifier checkpoints/classifier/best_model.pt \
  --output results/
```

## Configuration

Edit `configs/config.yaml` to customize:

- **Data parameters**: Sample rate, FFT size, mel bands, etc.
- **Model architecture**: Encoder dimensions, diffusion timesteps, etc.
- **Training parameters**: Batch size, learning rate, epochs, etc.
- **Feature extraction**: Which models to use (Wav2Vec2, HuBERT, OpenSMILE)

Example:
```yaml
data:
  target_sr: 16000
  n_mels: 80
  max_audio_length: 10

model:
  encoder:
    latent_dim: 256
  diffusion:
    num_timesteps: 1000

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0001
```

## Model Components

### 1. Audio Preprocessing
- Loads audio files and normalizes to [-1, 1]
- Converts to log mel spectrograms
- Pads/trims to fixed length

### 2. Semantic Feature Extraction
- **Wav2Vec2**: Self-supervised speech features
- **HuBERT**: Hidden unit BERT features
- **OpenSMILE**: Acoustic features (pitch, energy, MFCCs)

### 3. Encoder
- Compresses high-dimensional features to latent space
- Options: MLP, Transformer, Variational

### 4. Diffusion Model
- U-Net architecture with attention
- Conditioned on semantic features
- Generates mel spectrograms through reverse diffusion

### 5. Classifier
- CNN-based architecture
- Options: Basic CNN, ResNet, Attention-based
- Multi-label classification

## Pipeline Flow

```
Audio Input
    ↓
Preprocessing (Mel Spectrogram)
    ↓
Feature Extraction (Wav2Vec2/HuBERT/OpenSMILE)
    ↓
Encoder (Compress to Latent)
    ↓
Diffusion Model (Add/Remove Noise)
    ↓
Reconstructed Mel Spectrogram
    ↓
Classifier (Predict Mental Health)
    ↓
Output: Depression, Anxiety, Distress, Normal Probabilities
```

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

Navigate to `http://localhost:6006` to view:
- Loss curves
- Learning rate
- Validation metrics
- Generated samples

## Checkpoints

Models are saved to `checkpoints/` directory:
- `checkpoint_epoch{N}.pt`: Regular checkpoints every N epochs
- `best_model.pt`: Best model based on validation F1 score

## Outputs

Generated samples and reconstructed audio are saved to `outputs/`:
- `sample_epoch{N}_{i}.wav`: Samples from diffusion model
- `{filename}_reconstructed.wav`: Reconstructed audio
- `{filename}_original.wav`: Original audio

## Citation

If you use this code, please cite:
```
@article{mental_health_diffusion_2026,
  title={Diffusion Models for Mental Health Audio Analysis},
  author={Your Name},
  year={2026}
}
```

## License

MIT License

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `max_audio_length`
- Use gradient checkpointing

### Slow Training
- Enable feature caching: `cache_features=True` in dataset
- Reduce number of diffusion timesteps
- Use fewer feature extraction models

### Poor Classification Results
- Try different classifier architectures (ResNet, Attention)
- Increase training epochs
- Use data augmentation
- Collect more training data

### Audio Quality Issues
- Increase mel spectrogram resolution (`n_mels`)
- Use higher sample rate
- Use neural vocoder instead of Griffin-Lim

## Future Improvements

- [ ] Implement neural vocoder (HiFi-GAN/WaveGlow)
- [ ] Add data augmentation
- [ ] Implement classifier-free guidance
- [ ] Add emotion recognition
- [ ] Multi-modal analysis (text + audio)
- [ ] Real-time inference
- [ ] Model quantization for edge deployment
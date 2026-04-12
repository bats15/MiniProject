#!/usr/bin/env python3
"""
Quick start script to test the mental health audio pipeline
"""

import torch
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'transformers': 'Hugging Face Transformers',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies installed! ✓")
    return True


def test_preprocessing():
    """Test audio preprocessing"""
    print("\n" + "="*60)
    print("Testing Audio Preprocessing")
    print("="*60)
    
    from utils.preprocessing import AudioPreprocessor
    
    preprocessor = AudioPreprocessor()
    
    # Create dummy audio
    import numpy as np
    duration = 5
    sr = 16000
    audio = np.random.randn(sr * duration) * 0.1
    
    print(f"Input audio shape: {audio.shape}")
    
    # Convert to mel spectrogram
    mel_spec = preprocessor.audio_to_mel_spectrogram(audio)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    
    # Reconstruct audio
    reconstructed = preprocessor.mel_to_audio(mel_spec)
    print(f"Reconstructed audio shape: {reconstructed.shape}")
    
    print("✓ Preprocessing test passed!")


def test_feature_extraction():
    """Test feature extraction (without loading large models)"""
    print("\n" + "="*60)
    print("Testing Feature Extraction")
    print("="*60)
    
    from utils.feature_extraction import SemanticFeatureExtractor
    import numpy as np
    
    # Use basic features only (no large models)
    extractor = SemanticFeatureExtractor(
        use_wav2vec=False,
        use_hubert=False,
        use_opensmile=False
    )
    
    # Create dummy audio
    sr = 16000
    audio = np.random.randn(sr * 5) * 0.1
    
    print(f"Input audio shape: {audio.shape}")
    
    # Extract features
    features = extractor.extract_all_features(audio)
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature dimension: {len(features)}")
    
    print("✓ Feature extraction test passed!")


def test_models():
    """Test model instantiation"""
    print("\n" + "="*60)
    print("Testing Model Instantiation")
    print("="*60)
    
    from models.encoder import SemanticEncoder
    from models.diffusion import DiffusionUNet
    from models.classifier import MentalHealthClassifier
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test encoder
    print("\n1. Testing Encoder...")
    encoder = SemanticEncoder(input_dim=100, latent_dim=256)
    encoder.to(device)
    print(f"   Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test with dummy input
    x = torch.randn(4, 100).to(device)
    latent = encoder(x)
    print(f"   Input: {x.shape} -> Output: {latent.shape}")
    
    # Test diffusion model
    print("\n2. Testing Diffusion Model...")
    diffusion_model = DiffusionUNet()
    diffusion_model.to(device)
    print(f"   Parameters: {sum(p.numel() for p in diffusion_model.parameters()):,}")
    
    # Test with dummy input
    x = torch.randn(2, 1, 80, 313).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    condition = torch.randn(2, 256).to(device)
    output = diffusion_model(x, t, condition)
    print(f"   Input: {x.shape} -> Output: {output.shape}")
    
    # Test classifier
    print("\n3. Testing Classifier...")
    classifier = MentalHealthClassifier()
    classifier.to(device)
    print(f"   Parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # Test with dummy input
    x = torch.randn(4, 1, 80, 313).to(device)
    output = classifier(x)
    print(f"   Input: {x.shape} -> Output: {output.shape}")
    
    print("\n✓ All models instantiated successfully!")


def create_test_dataset():
    """Create a small test dataset"""
    print("\n" + "="*60)
    print("Creating Test Dataset")
    print("="*60)
    
    from create_dummy_data import create_dummy_dataset
    
    output_dir = "./data_test"
    num_samples = 20  # Small dataset for testing
    
    create_dummy_dataset(output_dir, num_samples=num_samples, duration=3)
    
    print(f"\n✓ Test dataset created in {output_dir}")
    return output_dir


def run_quick_test():
    """Run a quick end-to-end test"""
    print("\n" + "="*60)
    print("Running Quick End-to-End Test")
    print("="*60)
    
    from utils.preprocessing import AudioPreprocessor
    from utils.feature_extraction import SemanticFeatureExtractor
    from models.encoder import SemanticEncoder
    from models.classifier import MentalHealthClassifier
    import numpy as np
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Create dummy audio
    print("\n1. Creating dummy audio...")
    sr = 16000
    duration = 5
    audio = np.random.randn(sr * duration) * 0.1
    
    # 2. Preprocess
    print("2. Preprocessing...")
    preprocessor = AudioPreprocessor()
    mel_spec = preprocessor.audio_to_mel_spectrogram(audio)
    
    # 3. Extract features
    print("3. Extracting features...")
    extractor = SemanticFeatureExtractor(
        use_wav2vec=False, use_hubert=False, use_opensmile=False
    )
    features = extractor.extract_all_features(audio)
    
    # 4. Encode
    print("4. Encoding features...")
    encoder = SemanticEncoder(input_dim=len(features), latent_dim=256).to(device)
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    latent = encoder(features_tensor)
    
    # 5. Classify
    print("5. Classifying...")
    classifier = MentalHealthClassifier().to(device)
    mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
    logits = classifier(mel_tensor)
    probs = torch.sigmoid(logits)
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Audio shape: {audio.shape}")
    print(f"  Mel spec shape: {mel_spec.shape}")
    print(f"  Features dim: {features.shape}")
    print(f"  Latent dim: {latent.shape}")
    print(f"  Predictions: {probs.squeeze().tolist()}")
    print(f"{'='*60}")
    
    print("\n✓ Quick test completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Quick start script for mental health audio pipeline")
    parser.add_argument('--test', choices=['all', 'deps', 'preprocess', 'features', 'models', 'dataset', 'quick'],
                      default='all', help='Which test to run')
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Mental Health Audio Diffusion Model - Quick Start Script   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    tests = {
        'deps': check_dependencies,
        'preprocess': test_preprocessing,
        'features': test_feature_extraction,
        'models': test_models,
        'dataset': create_test_dataset,
        'quick': run_quick_test
    }
    
    if args.test == 'all':
        # Run all tests
        if not check_dependencies():
            return
        
        test_preprocessing()
        test_feature_extraction()
        test_models()
        create_test_dataset()
        run_quick_test()
    else:
        # Run specific test
        tests[args.test]()
    
    print(f"\n{'='*60}")
    print("All tests completed! 🎉")
    print("="*60)
    print("\nNext steps:")
    print("  1. Create a real dataset or use the test dataset")
    print("  2. Train the diffusion model: python train_diffusion.py")
    print("  3. Train the classifier: python train_classifier.py")
    print("  4. Run inference: python inference.py --audio <path>")
    print("\nSee README.md for detailed instructions.")


if __name__ == '__main__':
    main()
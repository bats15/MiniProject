"""
Create a dummy dataset for testing the pipeline
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import json
import argparse


def generate_dummy_audio(duration=5, sr=16000, seed=None):
    """
    Generate dummy audio with some structure
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        seed: Random seed
        
    Returns:
        Audio array
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate base signal
    t = np.linspace(0, duration, int(sr * duration))
    
    # Add some frequency components
    frequencies = [220, 440, 880]  # A3, A4, A5
    audio = np.zeros_like(t)
    
    for freq in frequencies:
        audio += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # Add noise
    audio += 0.05 * np.random.randn(len(t))
    
    # Add amplitude modulation
    audio *= (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio


def create_dummy_dataset(output_dir, num_samples=100, duration=5):
    """
    Create a complete dummy dataset with annotations
    
    Args:
        output_dir: Output directory
        num_samples: Total number of samples
        duration: Audio duration in seconds
    """
    output_path = Path(output_dir)
    sr = 16000
    
    # Define splits
    splits = {
        'train': int(num_samples * 0.7),
        'val': int(num_samples * 0.15),
        'test': int(num_samples * 0.15)
    }
    
    # Class names
    classes = ['depression', 'anxiety', 'distress', 'normal']
    
    print(f"Creating dummy dataset in {output_dir}")
    print(f"Total samples: {num_samples}")
    print(f"Splits: {splits}")
    
    for split, num in splits.items():
        print(f"\n=== Creating {split} split ({num} samples) ===")
        
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        for class_name in classes:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
        
        # Create annotations
        annotations = []
        
        samples_per_class = num // len(classes)
        
        for class_idx, class_name in enumerate(classes):
            class_dir = split_dir / class_name
            
            for i in range(samples_per_class):
                # Generate audio with different characteristics per class
                seed = hash(f"{split}_{class_name}_{i}") % (2**31)
                audio = generate_dummy_audio(duration=duration, sr=sr, seed=seed)
                
                # Modify based on class (just for variation)
                if class_name == 'depression':
                    audio *= 0.7  # Lower amplitude
                elif class_name == 'anxiety':
                    audio += 0.02 * np.random.randn(len(audio))  # More noise
                elif class_name == 'distress':
                    audio = np.clip(audio * 1.5, -1, 1)  # Higher amplitude, clipped
                
                # Save audio
                filename = f"{class_name}_{i:04d}.wav"
                filepath = class_dir / filename
                sf.write(filepath, audio, sr)
                
                # Create annotation
                labels = [0, 0, 0, 0]
                labels[class_idx] = 1
                
                annotations.append({
                    'audio_file': f"{split}/{class_name}/{filename}",
                    'labels': labels,
                    'metadata': {
                        'class': class_name,
                        'duration': duration,
                        'sample_rate': sr
                    }
                })
                
                if (i + 1) % 10 == 0:
                    print(f"  Created {i+1}/{samples_per_class} samples for {class_name}")
        
        # Save annotations
        annotation_file = output_path / f"{split}_annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"  Saved annotations to {annotation_file}")
    
    print(f"\n{'='*60}")
    print("Dataset creation complete!")
    print(f"Total audio files created: {num_samples}")
    print(f"Dataset location: {output_dir}")
    print("\nDataset structure:")
    print("  data/")
    print("    train/")
    print("      depression/")
    print("      anxiety/")
    print("      distress/")
    print("      normal/")
    print("    val/")
    print("      ...")
    print("    test/")
    print("      ...")
    print("    train_annotations.json")
    print("    val_annotations.json")
    print("    test_annotations.json")


def main():
    parser = argparse.ArgumentParser(description="Create dummy mental health audio dataset")
    parser.add_argument('--output', type=str, default='./data', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100, help='Total number of samples')
    parser.add_argument('--duration', type=int, default=5, help='Audio duration in seconds')
    
    args = parser.parse_args()
    
    create_dummy_dataset(args.output, args.num_samples, args.duration)


if __name__ == '__main__':
    main()
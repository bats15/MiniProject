"""
PyTorch Dataset for mental health audio data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from utils.preprocessing import AudioPreprocessor
from utils.feature_extraction import SemanticFeatureExtractor

class MentalHealthAudioDataset(Dataset):
    """Dataset for mental health audio with labels"""
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: AudioPreprocessor,
        feature_extractor: SemanticFeatureExtractor,
        split: str = 'train',
        cache_features: bool = True,
        label_map: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            data_dir: Directory containing audio files
            preprocessor: AudioPreprocessor instance
            feature_extractor: SemanticFeatureExtractor instance
            split: 'train', 'val', or 'test'
            cache_features: Whether to cache extracted features
            label_map: Mapping from label names to indices
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.split = split
        self.cache_features = cache_features
        
        # Default label map (Depression, Anxiety, Distress)
        if label_map is None:
            self.label_map = {
                'depression': 0,
                'anxiety': 1,
                'distress': 2,
                'normal': 3
            }
        else:
            self.label_map = label_map
        
        # Load data
        self.data = self._load_data()
        
        # Cache for features
        self.feature_cache = {}
        
        if cache_features:
            self._cache_all_features()
    
    def _load_data(self) -> List[Dict]:
        """Load audio files and labels"""
        data = []
        
        # Look for annotation file
        annotation_file = self.data_dir / f"{self.split}_annotations.json"
        
        if annotation_file.exists():
            # Load from annotation file
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            for item in annotations:
                audio_path = self.data_dir / item['audio_file']
                if audio_path.exists():
                    data.append({
                        'audio_path': str(audio_path),
                        'labels': item['labels'],  # Multi-label: [0, 1, 0]
                        'metadata': item.get('metadata', {})
                    })
        else:
            # Fallback: scan directory structure
            # Expected structure: data_dir/split/class_name/*.wav
            split_dir = self.data_dir / self.split
            
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name.lower()
                        if class_name in self.label_map:
                            label_idx = self.label_map[class_name]
                            
                            for audio_file in class_dir.glob("*.wav"):
                                # One-hot encoding
                                labels = [0] * len(self.label_map)
                                labels[label_idx] = 1
                                
                                data.append({
                                    'audio_path': str(audio_file),
                                    'labels': labels,
                                    'metadata': {'class': class_name}
                                })
        
        print(f"Loaded {len(data)} samples for {self.split} split")
        return data
    
    def _cache_all_features(self):
        """Pre-compute and cache all features"""
        print(f"Caching features for {self.split} split...")
        
        for idx in tqdm(range(len(self.data))):
            audio_path = self.data[idx]['audio_path']
            
            # Load and process audio
            audio = self.preprocessor.load_and_normalize(audio_path)
            mel_spec = self.preprocessor.audio_to_mel_spectrogram(audio)
            
            # Extract semantic features
            semantic_features = self.feature_extractor.extract_all_features(audio)
            
            # Cache
            self.feature_cache[idx] = {
                'mel_spec': mel_spec,
                'semantic_features': semantic_features
            }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary with keys:
                - mel_spec: (1, n_mels, time)
                - semantic_features: (feature_dim,)
                - labels: (num_classes,)
        """
        # Always keep path available for output metadata
        audio_path = self.data[idx]['audio_path']

        # Check cache first
        if idx in self.feature_cache:
            mel_spec = self.feature_cache[idx]['mel_spec']
            semantic_features = self.feature_cache[idx]['semantic_features']
        else:
            # Load and process
            audio = self.preprocessor.load_and_normalize(audio_path)
            mel_spec = self.preprocessor.audio_to_mel_spectrogram(audio)
            semantic_features = self.feature_extractor.extract_all_features(audio)
        
        # Get labels
        labels = np.array(self.data[idx]['labels'], dtype=np.float32)
        
        # Convert to tensors
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, n_mels, time)
        semantic_features_tensor = torch.FloatTensor(semantic_features)
        labels_tensor = torch.FloatTensor(labels)
        
        return {
            'mel_spec': mel_spec_tensor,
            'semantic_features': semantic_features_tensor,
            'labels': labels_tensor,
            'audio_path': audio_path
        }


def create_dataloaders(
    data_dir: str,
    preprocessor: AudioPreprocessor,
    feature_extractor: SemanticFeatureExtractor,
    batch_size: int = 16,
    num_workers: int = 4,
    cache_features: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory containing data splits
        preprocessor: AudioPreprocessor instance
        feature_extractor: SemanticFeatureExtractor instance
        batch_size: Batch size
        num_workers: Number of worker processes
        cache_features: Whether to cache features
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MentalHealthAudioDataset(
        data_dir, preprocessor, feature_extractor, 
        split='train', cache_features=cache_features
    )
    
    val_dataset = MentalHealthAudioDataset(
        data_dir, preprocessor, feature_extractor,
        split='val', cache_features=cache_features
    )
    
    test_dataset = MentalHealthAudioDataset(
        data_dir, preprocessor, feature_extractor,
        split='test', cache_features=cache_features
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_dummy_dataset(output_dir: str, num_samples: int = 100):
    """
    Create a dummy dataset for testing
    
    Args:
        output_dir: Directory to save dummy data
        num_samples: Number of samples to generate
    """
    import soundfile as sf
    
    output_path = Path(output_dir)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['depression', 'anxiety', 'distress', 'normal']:
            class_dir = output_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy audio files
    sr = 16000
    duration = 5  # seconds
    
    splits = {
        'train': int(num_samples * 0.7),
        'val': int(num_samples * 0.15),
        'test': int(num_samples * 0.15)
    }
    
    classes = ['depression', 'anxiety', 'distress', 'normal']
    
    for split, num in splits.items():
        samples_per_class = num // len(classes)
        
        for class_name in classes:
            class_dir = output_path / split / class_name
            
            for i in range(samples_per_class):
                # Generate random audio
                audio = np.random.randn(sr * duration) * 0.1
                
                # Save
                filename = f"{class_name}_{i:04d}.wav"
                sf.write(class_dir / filename, audio, sr)
    
    print(f"Created dummy dataset in {output_dir}")

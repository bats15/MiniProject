"""Compatibility shim for dataset utilities."""

from dataset import MentalHealthAudioDataset, create_dataloaders, create_dummy_dataset

__all__ = ["MentalHealthAudioDataset", "create_dataloaders", "create_dummy_dataset"]

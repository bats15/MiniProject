"""
Training script for the mental health classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models.classifier import MentalHealthClassifier, ResNetClassifier, AttentionClassifier
from utils.preprocessing import AudioPreprocessor
from utils.feature_extraction import SemanticFeatureExtractor
from utils.dataset import create_dataloaders


class ClassifierTrainer:
    """Trainer for the mental health classifier"""
    
    def __init__(self, config_path: str, model_type: str = 'basic'):
        """
        Args:
            config_path: Path to configuration file
            model_type: Type of classifier ('basic', 'resnet', 'attention')
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(
            self.config['device'] if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Create directories
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir']) / 'classifier'
        self.log_dir = Path(self.config['paths']['log_dir']) / 'classifier'
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            target_sr=self.config['data']['target_sr'],
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length'],
            n_mels=self.config['data']['n_mels'],
            max_length_sec=self.config['data']['max_audio_length']
        )
        
        # Initialize feature extractor
        self.feature_extractor = SemanticFeatureExtractor(
            device=self.device,
            use_wav2vec=False,  # Don't need for classifier
            use_hubert=False,
            use_opensmile=False
        )
        
        # Initialize model
        num_classes = self.config['model']['classifier']['num_classes']
        dropout = self.config['model']['classifier']['dropout']
        
        if model_type == 'basic':
            self.model = MentalHealthClassifier(
                input_channels=1,
                num_classes=num_classes,
                dropout=dropout
            ).to(self.device)
        elif model_type == 'resnet':
            self.model = ResNetClassifier(
                input_channels=1,
                num_classes=num_classes,
                dropout=dropout
            ).to(self.device)
        elif model_type == 'attention':
            self.model = AttentionClassifier(
                input_channels=1,
                num_classes=num_classes,
                dropout=dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Using {model_type} classifier")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss function (multi-label classification)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs']
        )
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_f1 = 0.0
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            mel_specs = batch['mel_spec'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward
            logits = self.model(mel_specs)
            loss = self.criterion(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            
            # Predictions
            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Log
            if self.global_step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            
            pbar.set_postfix({'loss': loss.item()})
            self.global_step += 1
        
        # Compute metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = self.compute_metrics(all_preds, all_labels)
        metrics['loss'] = epoch_loss / len(dataloader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Validating"):
            mel_specs = batch['mel_spec'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward
            logits = self.model(mel_specs)
            loss = self.criterion(logits, labels)
            
            val_loss += loss.item()
            
            # Predictions
            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        # Compute metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = self.compute_metrics(all_preds, all_labels)
        metrics['loss'] = val_loss / len(dataloader)
        
        return metrics
    
    def compute_metrics(self, preds, labels):
        """Compute classification metrics"""
        # Per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        # Overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        
        # Accuracy
        accuracy = accuracy_score(labels.flatten(), preds.flatten())
        
        return {
            'accuracy': accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"\nEpoch {epoch} - Train Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  F1: {train_metrics['f1']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            print(f"\nEpoch {epoch} - Val Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  F1: {val_metrics['f1']:.4f}")
            
            # Log to tensorboard
            for key, value in train_metrics.items():
                if not key.startswith('per_class'):
                    self.writer.add_scalar(f'train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                if not key.startswith('per_class'):
                    self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
            
            if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
                self.save_checkpoint(is_best)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model', type=str, default='basic', choices=['basic', 'resnet', 'attention'])
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # Create trainer
    trainer = ClassifierTrainer(args.config, args.model)
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=trainer.config['paths']['data_dir'],
        preprocessor=trainer.preprocessor,
        feature_extractor=trainer.feature_extractor,
        batch_size=trainer.config['training']['batch_size'],
        num_workers=trainer.config['training']['num_workers']
    )
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
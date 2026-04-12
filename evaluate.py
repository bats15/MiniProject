"""
Evaluation script for mental health audio models
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import json

from models.classifier import MentalHealthClassifier
from utils.preprocessing import AudioPreprocessor
from utils.feature_extraction import SemanticFeatureExtractor
from utils.dataset import create_dataloaders


class ModelEvaluator:
    """Evaluate mental health classifier"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            config_path: Path to config file
            checkpoint_path: Path to model checkpoint
            device: Device to use
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize preprocessor and feature extractor
        self.preprocessor = AudioPreprocessor(
            target_sr=self.config['data']['target_sr'],
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length'],
            n_mels=self.config['data']['n_mels'],
            max_length_sec=self.config['data']['max_audio_length']
        )
        
        self.feature_extractor = SemanticFeatureExtractor(
            device=self.device,
            use_wav2vec=False,
            use_hubert=False,
            use_opensmile=False
        )
        
        # Load model
        self.model = MentalHealthClassifier(
            input_channels=1,
            num_classes=self.config['model']['classifier']['num_classes']
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Class names
        self.class_names = ['Depression', 'Anxiety', 'Distress', 'Normal']
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        Evaluate model on dataloader
        
        Args:
            dataloader: DataLoader to evaluate on
            
        Returns:
            Dictionary with metrics and predictions
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            mel_specs = batch['mel_spec'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(mel_specs)
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        # Concatenate all batches
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        all_probs = np.vstack(all_probs)
        
        # Compute metrics
        metrics = self.compute_metrics(all_preds, all_labels, all_probs)
        
        return {
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def compute_metrics(self, preds, labels, probs):
        """Compute comprehensive metrics"""
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(labels.flatten(), preds.flatten())
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels[:, i], preds[:, i], average='binary', zero_division=0
            )
            
            metrics[f'{class_name.lower()}_precision'] = precision
            metrics[f'{class_name.lower()}_recall'] = recall
            metrics[f'{class_name.lower()}_f1'] = f1
            
            # AUC if possible
            if len(np.unique(labels[:, i])) > 1:
                auc = roc_auc_score(labels[:, i], probs[:, i])
                metrics[f'{class_name.lower()}_auc'] = auc
        
        # Macro-averaged metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        
        metrics['macro_precision'] = precision
        metrics['macro_recall'] = recall
        metrics['macro_f1'] = f1
        
        # Micro-averaged metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='micro', zero_division=0
        )
        
        metrics['micro_precision'] = precision
        metrics['micro_recall'] = recall
        metrics['micro_f1'] = f1
        
        return metrics
    
    def plot_confusion_matrix(self, labels, preds, output_path):
        """Plot confusion matrix for each class"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(labels[:, i], preds[:, i])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[i]
            )
            
            axes[i].set_title(f'{class_name} Confusion Matrix')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
        plt.close()
    
    def plot_roc_curves(self, labels, probs, output_path):
        """Plot ROC curves for each class"""
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            if len(np.unique(labels[:, i])) > 1:
                fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {output_path}")
        plt.close()
    
    def plot_class_distribution(self, labels, output_path):
        """Plot distribution of classes in dataset"""
        class_counts = labels.sum(axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.class_names, class_counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution in Dataset')
        plt.xticks(rotation=45)
        
        for i, count in enumerate(class_counts):
            plt.text(i, count, str(int(count)), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution to {output_path}")
        plt.close()
    
    def save_results(self, results, output_dir):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        print(f"Saved metrics to {metrics_file}")
        
        # Save predictions
        np.save(output_path / 'predictions.npy', results['predictions'])
        np.save(output_path / 'labels.npy', results['labels'])
        np.save(output_path / 'probabilities.npy', results['probabilities'])
        
        # Generate plots
        self.plot_confusion_matrix(
            results['labels'], 
            results['predictions'],
            output_path / 'confusion_matrix.png'
        )
        
        self.plot_roc_curves(
            results['labels'],
            results['probabilities'],
            output_path / 'roc_curves.png'
        )
        
        self.plot_class_distribution(
            results['labels'],
            output_path / 'class_distribution.png'
        )
        
        # Print classification report
        print("\n" + "="*60)
        print("Classification Report")
        print("="*60)
        
        for i, class_name in enumerate(self.class_names):
            print(f"\n{class_name}:")
            print(classification_report(
                results['labels'][:, i],
                results['predictions'][:, i],
                target_names=['Negative', 'Positive']
            ))


def main():
    parser = argparse.ArgumentParser(description="Evaluate mental health classifier")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='evaluation_results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.config, args.checkpoint, args.device)
    
    # Create dataloaders
    _, _, test_loader = create_dataloaders(
        data_dir=evaluator.config['paths']['data_dir'],
        preprocessor=evaluator.preprocessor,
        feature_extractor=evaluator.feature_extractor,
        batch_size=evaluator.config['training']['batch_size'],
        num_workers=evaluator.config['training']['num_workers']
    )
    
    # Evaluate
    print("Evaluating model on test set...")
    results = evaluator.evaluate(test_loader)
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
    
    # Save results
    evaluator.save_results(results, args.output)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
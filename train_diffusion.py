"""
Training script for the diffusion model
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

from models.encoder import SemanticEncoder
from models.diffusion import DiffusionUNet, GaussianDiffusion
from utils.preprocessing import AudioPreprocessor
from utils.feature_extraction import SemanticFeatureExtractor
from utils.dataset import create_dataloaders


class DiffusionTrainer:
    """Trainer for the diffusion model"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to configuration file
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
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessors
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
            use_wav2vec=self.config['features']['use_wav2vec'],
            use_hubert=self.config['features']['use_hubert'],
            use_opensmile=self.config['features']['use_opensmile']
        )
        
        # Get feature dimension
        feature_dim = self.feature_extractor.get_feature_dim()
        print(f"Feature dimension: {feature_dim}")
        
        # Update config
        self.config['model']['encoder']['input_dim'] = feature_dim
        
        # Initialize models
        self.encoder = SemanticEncoder(
            input_dim=feature_dim,
            latent_dim=self.config['model']['encoder']['latent_dim'],
            hidden_dims=self.config['model']['encoder']['hidden_dims'],
            dropout=self.config['model']['encoder']['dropout']
        ).to(self.device)
        
        self.diffusion_model = DiffusionUNet(
            in_channels=self.config['model']['diffusion']['in_channels'],
            condition_dim=self.config['model']['diffusion']['condition_dim'],
            time_dim=self.config['model']['diffusion']['time_dim'],
            base_channels=self.config['model']['diffusion']['base_channels']
        ).to(self.device)
        
        # Initialize diffusion process
        self.diffusion = GaussianDiffusion(
            num_timesteps=self.config['model']['diffusion']['num_timesteps'],
            beta_start=self.config['model']['diffusion']['beta_start'],
            beta_end=self.config['model']['diffusion']['beta_end']
        )
        
        # Move diffusion parameters to device
        self.diffusion.betas = self.diffusion.betas.to(self.device)
        self.diffusion.alphas = self.diffusion.alphas.to(self.device)
        self.diffusion.alphas_cumprod = self.diffusion.alphas_cumprod.to(self.device)
        self.diffusion.sqrt_alphas_cumprod = self.diffusion.sqrt_alphas_cumprod.to(self.device)
        self.diffusion.sqrt_one_minus_alphas_cumprod = self.diffusion.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.diffusion.sqrt_recip_alphas = self.diffusion.sqrt_recip_alphas.to(self.device)
        self.diffusion.posterior_variance = self.diffusion.posterior_variance.to(self.device)
        
        # Optimizers
        self.optimizer = AdamW(
            list(self.encoder.parameters()) + list(self.diffusion_model.parameters()),
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
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.encoder.train()
        self.diffusion_model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            mel_specs = batch['mel_spec'].to(self.device)
            semantic_features = batch['semantic_features'].to(self.device)
            
            # Encode semantic features
            latent = self.encoder(semantic_features)
            
            # Sample random timesteps
            batch_size = mel_specs.shape[0]
            t = torch.randint(
                0, self.diffusion.num_timesteps, 
                (batch_size,), 
                device=self.device
            )
            
            # Add noise
            noise = torch.randn_like(mel_specs)
            noisy_specs = self.diffusion.q_sample(mel_specs, t, noise)
            
            # Predict noise
            predicted_noise = self.diffusion_model(noisy_specs, t, latent)
            
            # Loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.diffusion_model.parameters()),
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log
            if self.global_step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            pbar.set_postfix({'loss': loss.item()})
            self.global_step += 1
        
        return epoch_loss / num_batches
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validate the model"""
        self.encoder.eval()
        self.diffusion_model.eval()
        
        val_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Validating"):
            mel_specs = batch['mel_spec'].to(self.device)
            semantic_features = batch['semantic_features'].to(self.device)
            
            # Encode
            latent = self.encoder(semantic_features)
            
            # Sample timesteps
            batch_size = mel_specs.shape[0]
            t = torch.randint(
                0, self.diffusion.num_timesteps,
                (batch_size,),
                device=self.device
            )
            
            # Add noise
            noise = torch.randn_like(mel_specs)
            noisy_specs = self.diffusion.q_sample(mel_specs, t, noise)
            
            # Predict
            predicted_noise = self.diffusion_model(noisy_specs, t, latent)
            
            # Loss
            loss = F.mse_loss(predicted_noise, noise)
            
            val_loss += loss.item()
            num_batches += 1
        
        return val_loss / num_batches
    
    @torch.no_grad()
    def generate_samples(self, num_samples=4):
        """Generate sample audio"""
        self.encoder.eval()
        self.diffusion_model.eval()
        
        # Use random conditions
        condition = torch.randn(
            num_samples, 
            self.config['model']['encoder']['latent_dim'],
            device=self.device
        )
        
        # Generate
        shape = (num_samples, 1, self.config['data']['n_mels'], 313)  # Adjust time dimension
        generated = self.diffusion.p_sample_loop(
            self.diffusion_model,
            shape,
            condition,
            self.device
        )
        
        # Convert to audio
        for i in range(num_samples):
            mel = generated[i, 0].cpu().numpy()
            audio = self.preprocessor.mel_to_audio(mel)
            
            # Save
            output_path = self.output_dir / f"sample_epoch{self.epoch}_{i}.wav"
            self.preprocessor.save_audio(audio, str(output_path))
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'diffusion_model_state_dict': self.diffusion_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
            
            # Log
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
                self.save_checkpoint()
                self.generate_samples()
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # Create trainer
    trainer = DiffusionTrainer(args.config)
    
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
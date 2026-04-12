"""
Complete inference pipeline for mental health audio analysis
"""

import torch
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, Optional

from models.encoder import SemanticEncoder
from models.diffusion import DiffusionUNet, GaussianDiffusion
from models.classifier import MentalHealthClassifier
from utils.preprocessing import AudioPreprocessor
from utils.feature_extraction import SemanticFeatureExtractor


class MentalHealthAudioPipeline:
    """Complete pipeline for mental health audio analysis"""
    
    def __init__(
        self,
        config_path: str,
        encoder_checkpoint: Optional[str] = None,
        diffusion_checkpoint: Optional[str] = None,
        classifier_checkpoint: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            config_path: Path to configuration file
            encoder_checkpoint: Path to encoder checkpoint
            diffusion_checkpoint: Path to diffusion model checkpoint
            classifier_checkpoint: Path to classifier checkpoint
            device: Device to run inference on
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
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
            device='cpu',
            use_wav2vec=self.config['features']['use_wav2vec'],
            use_hubert=self.config['features']['use_hubert'],
            use_opensmile=self.config['features']['use_opensmile']
        )
        
        # Get feature dimension
        feature_dim = self.feature_extractor.get_feature_dim()
        
        # Initialize encoder
        self.encoder = SemanticEncoder(
            input_dim=feature_dim,
            latent_dim=self.config['model']['encoder']['latent_dim'],
            hidden_dims=self.config['model']['encoder']['hidden_dims']
        ).to(self.device)
        
        # Initialize diffusion model
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
        
        # Initialize classifier
        self.classifier = MentalHealthClassifier(
            input_channels=1,
            num_classes=self.config['model']['classifier']['num_classes']
        ).to(self.device)
        
        # Load checkpoints
        if encoder_checkpoint and diffusion_checkpoint:
            self.load_diffusion_checkpoint(encoder_checkpoint, diffusion_checkpoint)
        
        if classifier_checkpoint:
            self.load_classifier_checkpoint(classifier_checkpoint)
        
        # Set to eval mode
        self.encoder.eval()
        self.diffusion_model.eval()
        self.classifier.eval()
        
        # Class names
        self.class_names = ['Depression', 'Anxiety', 'Distress', 'Normal']
    
    def load_diffusion_checkpoint(self, encoder_path: str, diffusion_path: str):
        """Load encoder and diffusion model checkpoints"""
        # Can be same checkpoint or separate
        checkpoint = torch.load(encoder_path, map_location=self.device)
        
        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded diffusion checkpoint from {encoder_path}")
    
    def load_classifier_checkpoint(self, checkpoint_path: str):
        """Load classifier checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded classifier checkpoint from {checkpoint_path}")
    
    @torch.no_grad()
    def process_audio(self, audio_path: str, return_reconstructed: bool = True) -> Dict:
        """
        Process audio file through the complete pipeline
        
        Args:
            audio_path: Path to audio file
            return_reconstructed: Whether to reconstruct audio via diffusion
            
        Returns:
            Dictionary with results
        """
        # 1. Load and preprocess audio
        audio = self.preprocessor.load_and_normalize(audio_path)
        mel_spec = self.preprocessor.audio_to_mel_spectrogram(audio)
        
        # 2. Extract semantic features
        semantic_features = self.feature_extractor.extract_all_features(audio)
        semantic_tensor = torch.FloatTensor(semantic_features).unsqueeze(0).to(self.device)
        
        # 3. Encode to latent space
        latent = self.encoder(semantic_tensor)
        
        results = {
            'original_audio': audio,
            'original_mel': mel_spec,
            'latent_representation': latent.cpu().numpy()
        }
        
        # 4. Generate mel spectrogram via diffusion (optional)
        if return_reconstructed:
            mel_shape = (1, 1, self.config['data']['n_mels'], mel_spec.shape[1])
            generated_mel = self.diffusion.p_sample_loop(
                self.diffusion_model,
                mel_shape,
                latent,
                self.device
            )
            
            # Convert to audio
            mel_np = generated_mel.squeeze().cpu().numpy()
            reconstructed_audio = self.preprocessor.mel_to_audio(mel_np)
            
            results['reconstructed_mel'] = mel_np
            results['reconstructed_audio'] = reconstructed_audio
        else:
            # Use original mel for classification
            generated_mel = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 5. Classify mental health state
        logits = self.classifier(generated_mel)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Create results dictionary
        predictions = {}
        for i, class_name in enumerate(self.class_names):
            predictions[class_name.lower()] = float(probabilities[i])
        
        results['predictions'] = predictions
        results['dominant_class'] = self.class_names[np.argmax(probabilities)]
        
        return results
    
    def analyze_audio(self, audio_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze audio file and optionally save outputs
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save outputs (optional)
            
        Returns:
            Analysis results
        """
        print(f"Analyzing audio: {audio_path}")
        
        # Process
        results = self.process_audio(audio_path, return_reconstructed=True)
        
        # Print results
        print("\n=== Mental Health Analysis Results ===")
        print(f"Dominant Class: {results['dominant_class']}")
        print("\nProbabilities:")
        for class_name, prob in results['predictions'].items():
            print(f"  {class_name.capitalize()}: {prob:.4f}")
        
        # Save outputs
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save reconstructed audio
            if 'reconstructed_audio' in results:
                audio_output = output_path / f"{Path(audio_path).stem}_reconstructed.wav"
                self.preprocessor.save_audio(results['reconstructed_audio'], str(audio_output))
                print(f"\nSaved reconstructed audio to: {audio_output}")
            
            # Save original audio
            original_output = output_path / f"{Path(audio_path).stem}_original.wav"
            self.preprocessor.save_audio(results['original_audio'], str(original_output))
            print(f"Saved original audio to: {original_output}")
        
        return results
    
    def batch_analyze(self, audio_dir: str, output_dir: Optional[str] = None) -> list:
        """
        Analyze all audio files in a directory
        
        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save outputs (optional)
            
        Returns:
            List of results for each file
        """
        audio_files = list(Path(audio_dir).glob("*.wav")) + list(Path(audio_dir).glob("*.mp3"))
        
        all_results = []
        
        for audio_file in audio_files:
            print(f"\n{'='*60}")
            results = self.analyze_audio(str(audio_file), output_dir)
            results['filename'] = audio_file.name
            all_results.append(results)
        
        return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mental Health Audio Analysis Pipeline")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file or directory')
    parser.add_argument('--encoder', type=str, help='Path to encoder checkpoint')
    parser.add_argument('--diffusion', type=str, help='Path to diffusion checkpoint')
    parser.add_argument('--classifier', type=str, required=True, help='Path to classifier checkpoint')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = MentalHealthAudioPipeline(
        config_path=args.config,
        encoder_checkpoint=args.encoder,
        diffusion_checkpoint=args.diffusion,
        classifier_checkpoint=args.classifier,
        device=args.device
    )
    
    # Process
    if args.batch:
        results = pipeline.batch_analyze(args.audio, args.output)
    else:
        results = pipeline.analyze_audio(args.audio, args.output)


if __name__ == '__main__':
    main()
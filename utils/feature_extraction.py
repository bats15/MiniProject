"""
Semantic feature extraction using pretrained models
"""

import torch
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor, HubertModel
import warnings

warnings.filterwarnings('ignore')


class SemanticFeatureExtractor:
    """Extract semantic features from audio using pretrained models"""
    
    def __init__(
        self, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_wav2vec=True,
        use_hubert=False,
        use_opensmile=True,
        wav2vec_model="facebook/wav2vec2-base",
        hubert_model="facebook/hubert-base-ls960"
    ):
        """
        Args:
            device: Device to run models on
            use_wav2vec: Whether to use Wav2Vec2
            use_hubert: Whether to use HuBERT
            use_opensmile: Whether to use OpenSMILE features
            wav2vec_model: Wav2Vec2 model name
            hubert_model: HuBERT model name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wav2vec = use_wav2vec
        self.use_hubert = use_hubert
        self.use_opensmile = use_opensmile
        
        # Load Wav2Vec2
        if use_wav2vec:
            print("Loading Wav2Vec2 model...")
            try:
                self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model)
                self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model).to(self.device)
                self.wav2vec.eval()
            except Exception as e:
                warnings.warn(
                    f"Failed to load Wav2Vec2 model '{wav2vec_model}'. "
                    f"Disabling Wav2Vec2 features. Error: {e}"
                )
                self.use_wav2vec = False
        
        # Load HuBERT
        if use_hubert:
            print("Loading HuBERT model...")
            try:
                self.hubert = HubertModel.from_pretrained(hubert_model).to(self.device)
                self.hubert.eval()
            except Exception as e:
                warnings.warn(
                    f"Failed to load HuBERT model '{hubert_model}'. "
                    f"Disabling HuBERT features. Error: {e}"
                )
                self.use_hubert = False
        
        # Load OpenSMILE
        if use_opensmile:
            try:
                import opensmile
                print("Loading OpenSMILE...")
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.ComParE_2016,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
            except ImportError:
                print("OpenSMILE not available, skipping...")
                self.use_opensmile = False
    
    def extract_wav2vec_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract features using Wav2Vec2
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Feature vector (pooled over time)
        """
        if not self.use_wav2vec:
            return np.array([])
        
        # Process audio
        inputs = self.wav2vec_processor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.wav2vec(input_values)
            # Get last hidden state: (batch, time, hidden_dim)
            hidden_states = outputs.last_hidden_state
            
            # Pool over time dimension (mean pooling)
            pooled = torch.mean(hidden_states, dim=1)  # (batch, hidden_dim)
        
        return pooled.cpu().numpy().flatten()
    
    def extract_hubert_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract features using HuBERT
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Feature vector (pooled over time)
        """
        if not self.use_hubert:
            return np.array([])
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.hubert(audio_tensor)
            hidden_states = outputs.last_hidden_state
            
            # Pool over time
            pooled = torch.mean(hidden_states, dim=1)
        
        return pooled.cpu().numpy().flatten()
    
    def extract_opensmile_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract acoustic features using OpenSMILE
        Features include: pitch, speech rate, emotional tone, energy, etc.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            OpenSMILE feature vector
        """
        if not self.use_opensmile:
            return np.array([])
        
        try:
            features = self.smile.process_signal(audio, sr)
            return features.values.flatten()
        except Exception as e:
            print(f"OpenSMILE extraction failed: {e}")
            return np.zeros(6373)  # ComParE_2016 feature size
    
    def extract_basic_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract basic acoustic features as fallback
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Basic feature vector
        """
        import librosa
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Aggregate statistics
        features = np.concatenate([
            [np.mean(zcr), np.std(zcr)],
            [np.mean(spectral_centroid), np.std(spectral_centroid)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ])
        
        return features
    
    def extract_all_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Combine all semantic features
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Combined feature vector
        """
        features = []
        
        # Wav2Vec2 features
        if self.use_wav2vec:
            wav2vec_feat = self.extract_wav2vec_features(audio, sr)
            features.append(wav2vec_feat)
        
        # HuBERT features
        if self.use_hubert:
            hubert_feat = self.extract_hubert_features(audio, sr)
            features.append(hubert_feat)
        
        # OpenSMILE features
        if self.use_opensmile:
            opensmile_feat = self.extract_opensmile_features(audio, sr)
            features.append(opensmile_feat)
        
        # If no features were extracted, use basic features
        if not features:
            basic_feat = self.extract_basic_features(audio, sr)
            features.append(basic_feat)
        
        # Concatenate all features
        combined = np.concatenate(features)
        
        return combined
    
    def get_feature_dim(self) -> int:
        """Get the dimension of the feature vector"""
        # Create dummy audio to compute feature dimension
        dummy_audio = np.random.randn(16000)
        features = self.extract_all_features(dummy_audio)
        return len(features)

"""
Semantic feature encoder models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticEncoder(nn.Module):
    """MLP-based encoder for semantic features"""
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int = 256, 
        hidden_dims: list = [512, 384],
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent representation
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        # Final projection to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input features (batch, input_dim)
            
        Returns:
            Latent representation (batch, latent_dim)
        """
        return self.encoder(x)


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for sequential features"""
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int = 256, 
        nhead: int = 8, 
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent representation
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, latent_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(latent_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        """
        Args:
            x: Input features (batch, seq_len, input_dim) or (batch, input_dim)
            
        Returns:
            Latent representation (batch, latent_dim)
        """
        # If input is 2D, add sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Embed
        x = self.embedding(x)  # (batch, seq_len, latent_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)  # (batch, seq_len, latent_dim)
        
        # Pool over sequence dimension
        x = x.transpose(1, 2)  # (batch, latent_dim, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch, latent_dim)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class VariationalEncoder(nn.Module):
    """Variational autoencoder for semantic features"""
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int = 256, 
        hidden_dims: list = [512, 384]
    ):
        """
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent representation
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder network
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, return_dist=False):
        """
        Args:
            x: Input features (batch, input_dim)
            return_dist: Whether to return mu and logvar
            
        Returns:
            Latent representation (batch, latent_dim)
            or (z, mu, logvar) if return_dist=True
        """
        h = self.encoder(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        z = self.reparameterize(mu, logvar)
        
        if return_dist:
            return z, mu, logvar
        return z
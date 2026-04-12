"""
Diffusion model for audio generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: (batch,) timestep indices
            
        Returns:
            (batch, dim) embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        return h


class AttentionBlock(nn.Module):
    """Self-attention block"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # QKV
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        
        # Attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v
        
        # Reshape back
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        
        # Project and residual
        out = self.proj(out)
        return x + out


class DiffusionUNet(nn.Module):
    """U-Net architecture for diffusion model"""
    
    def __init__(
        self, 
        in_channels: int = 1,
        condition_dim: int = 256,
        time_dim: int = 256,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        use_attention: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels
            condition_dim: Dimension of conditioning vector
            time_dim: Dimension of time embedding
            base_channels: Base number of channels
            channel_mult: Channel multipliers for each resolution
            num_res_blocks: Number of residual blocks per resolution
            use_attention: Whether to use attention blocks
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.time_dim = time_dim
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Condition embedding
        self.cond_proj = nn.Linear(condition_dim, time_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        level_channels = [base_channels * mult for mult in channel_mult]
        curr_ch = base_channels
        self.skip_channels = []

        for i, out_ch in enumerate(level_channels):
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(curr_ch, out_ch, time_dim, use_attention and i >= 2)
                )
                curr_ch = out_ch
                self.skip_channels.append(out_ch)

            if i < len(level_channels) - 1:
                self.encoder_pools.append(nn.Conv2d(curr_ch, curr_ch, 3, stride=2, padding=1))

        # Bottleneck
        self.bottleneck = ResBlock(curr_ch, curr_ch, time_dim, use_attention=True)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        skip_channels = list(self.skip_channels)
        for level in reversed(range(len(level_channels))):
            out_ch = level_channels[level]

            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                self.decoder_blocks.append(
                    ResBlock(curr_ch + skip_ch, out_ch, time_dim, use_attention and level >= 2)
                )
                curr_ch = out_ch

            if level > 0:
                up_ch = level_channels[level - 1]
                self.decoder_upsamples.append(
                    nn.ConvTranspose2d(curr_ch, up_ch, 4, stride=2, padding=1)
                )
                curr_ch = up_ch
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, curr_ch),
            nn.GELU(),
            nn.Conv2d(curr_ch, in_channels, 3, padding=1)
        )
    
    def forward(self, x, t, condition):
        """
        Args:
            x: Noisy input (batch, in_channels, H, W)
            t: Timestep (batch,)
            condition: Conditioning vector (batch, condition_dim)
            
        Returns:
            Predicted noise (batch, in_channels, H, W)
        """
        # Time and condition embedding
        t_emb = self.time_mlp(t)
        c_emb = self.cond_proj(condition)
        emb = t_emb + c_emb
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder
        skip_connections = []
        encoder_idx = 0
        pool_idx = 0

        for i, _ in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                x = self.encoder_blocks[encoder_idx](x, emb)
                skip_connections.append(x)
                encoder_idx += 1

            if i < len(self.channel_mult) - 1:
                x = self.encoder_pools[pool_idx](x)
                pool_idx += 1
        
        # Bottleneck
        x = self.bottleneck(x, emb)
        
        # Decoder
        decoder_idx = 0
        upsample_idx = 0
        
        for i in reversed(range(len(self.channel_mult))):
            for _ in range(self.num_res_blocks):
                skip = skip_connections.pop()
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
                x = torch.cat([x, skip], dim=1)
                x = self.decoder_blocks[decoder_idx](x, emb)
                decoder_idx += 1

            if i > 0:
                x = self.decoder_upsamples[upsample_idx](x)
                upsample_idx += 1
        
        # Final conv
        return self.final_conv(x)


class ResBlock(nn.Module):
    """Residual block with time embedding"""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, use_attention: bool = False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.time_proj = nn.Linear(time_dim, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.activation = nn.GELU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(out_channels)
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # Add time embedding
        time_out = self.time_proj(time_emb)
        h = h + time_out[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        # Residual connection
        h = h + self.shortcut(x)
        
        # Attention
        if self.use_attention:
            h = self.attention(h)
        
        return h


class GaussianDiffusion:
    """Gaussian diffusion process"""
    
    def __init__(
        self, 
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = 'linear'
    ):
        """
        Args:
            num_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule: Noise schedule ('linear', 'cosine')
        """
        self.num_timesteps = num_timesteps
        
        # Create beta schedule
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Pre-compute values
        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule from Improved DDPM paper"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: add noise to clean data
        
        Args:
            x_start: Clean data (batch, C, H, W)
            t: Timestep (batch,)
            noise: Optional pre-generated noise
            
        Returns:
            Noisy data
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alpha = sqrt_alpha[:, None, None, None]
        sqrt_one_minus_alpha = sqrt_one_minus_alpha[:, None, None, None]
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    @torch.no_grad()
    def p_sample(self, model, x, t, condition):
        """
        Single reverse diffusion step
        
        Args:
            model: Denoising model
            x: Noisy data at timestep t
            t: Current timestep
            condition: Conditioning vector
            
        Returns:
            Denoised data at timestep t-1
        """
        # Predict noise
        predicted_noise = model(x, t, condition)
        
        # Get parameters
        alpha = self.alphas[t][:, None, None, None]
        alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
        beta = self.betas[t][:, None, None, None]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]
        
        # Compute x_{t-1}
        model_mean = sqrt_recip_alpha * (x - beta * predicted_noise / torch.sqrt(1 - alpha_cumprod))
        
        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_variance = self.posterior_variance[t][:, None, None, None]
            return model_mean + torch.sqrt(posterior_variance) * noise
        else:
            return model_mean
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition, device):
        """
        Complete reverse diffusion process
        
        Args:
            model: Denoising model
            shape: Shape of output (batch, C, H, W)
            condition: Conditioning vector
            device: Device
            
        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, condition)
        
        return x

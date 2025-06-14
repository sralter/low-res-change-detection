"""
vae.py

Variational Autoencoder (VAE) architectures for change-detection patches.

Contains:
  - VAEEncoder / VAEDecoder: classic VAE with fully-connected latent bottleneck
  - TimeEmbedding: MLP to encode Δt into latent space
  - VAE: wrapper combining encoder, reparameterization, decoder, and loss
  - SEBlock: squeeze-and-excitation channel attention
  - FCVAEEncoder / FCVAEDecoder: fully-convolutional VAE with skip connections
  - FCVAE: U-Net style VAE producing spatially structured latent maps

Supports:
  - optional time embeddings
  - slow-feature regularization
  - configurable latent dimension and input channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class VAEEncoder(nn.Module):
    """
    A VAE tries to learn a compressed (i.e., latent) representation of an input.
    Unlike a regular autoencoder, it learns not a point, but a distribution (mean + variance).
    This adds regularity and forces the latent space to be structured and generative.

    Other notes:
        stride=2 halves resolution each layer
        BatchNorm helps stabilize training
        ReLU is activation function
        Output spatial resolution is 8x8, giving 128x compression (128x128 -> 8x8)
        Latent dimension (64) can be changed easily

    Encoder module for a VAE. Applies a series of Conv2d layers with stride=2 to
    downsample the input and increase channel depth, then flattens to produce
    two vectors: mean (μ) and log-variance (logvar) for the latent distribution.

    Args:
        in_channels (int): Number of input channels (e.g., raw bands + metadata).
        latent_dim (int): Dimension of the latent space.
        
    """
    def __init__(self, 
                 in_channels=9+4, # raw bands + delta t + DOY + seasonal NDVI + z-score NDVI 
                 latent_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Convolutional layers: reduce spatial dims, increase depth
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Latent layers — compress to mean and logvar
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor of shape (B, in_channels, 128, 128).
        Returns:
            mu (Tensor): Latent means of shape (B, latent_dim).
            logvar (Tensor): Latent log-variances of shape (B, latent_dim).
        """
        # x: (batch_size, in_channels, H=128, W=128)
        x = self.encoder(x)             # → (B, 256, 8, 8)
        x_flat = self.flatten(x)       # → (B, 256*8*8)
        mu = self.fc_mu(x_flat)        # → (B, latent_dim)
        logvar = self.fc_logvar(x_flat)

        return mu, logvar
    
class VAEDecoder(nn.Module):
    """
    Converts a latent vector z of shape (B, latent_dim, sampled using mu and logvar) to:
        a reconstructed patch of shape (B, 9, 128, 128) to match original 9-band input
        fc: Expands the compressed latent vector into a spatial feature map
        ConvTranspose2d: Upsamples spatially (the reverse of Conv2d)
        Sigmoid(): Clamps outputs between 0-1 (works well for normalized inputs)

    Decoder module for a VAE. Expands a latent vector into a reconstructed patch
    via a fully-connected layer followed by ConvTranspose2d upsampling.

    Args:
        out_channels (int): Number of output channels (must match encoder input).
        latent_dim (int): Dimension of the latent space.

    """
    def __init__(self, out_channels=9, latent_dim=64):
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        # Project latent vector back to feature map shape (e.g., 256 × 8 × 8)
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        # Deconvolution / Upsampling path
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Sigmoid(),  # Restrict output between 0 and 1 (assuming normalized inputs)
        )


    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass through the decoder.

        Args:
            z (Tensor): Latent tensor of shape (B, latent_dim).
        Returns:
            recon (Tensor): Reconstructed output of shape (B, out_channels, 128, 128).
        """
        # z: (batch_size, latent_dim)
        x = self.fc(z)                 # (B, 256*8*8)
        x = x.view(-1, 256, 8, 8)      # (B, 256, 8, 8)
        x = self.decoder(x)           # (B, out_channels, 128, 128)
        return x

class TimeEmbedding(nn.Module):
    """
    MLP that encodes a single scalar Δt into a latent vector of size latent_dim.

    Args:
        latent_dim (int): Size of the output embedding.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, delta_t: Tensor) -> Tensor:
        """
        Forward pass through the time embedding.

        Args:
            delta_t (Tensor): Δt tensor of shape (B, 1).
        Returns:
            Tensor: Embedding of shape (B, latent_dim).
        """
        # delta_t: (B, 1)
        return self.mlp(delta_t)

# --- VAE wrapper class

class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder, reparameterization trick, optional
    time embedding, and decoder.

    Args:
        in_channels (int): Number of input channels.
        latent_dim (int): Latent dimensionality.
        use_time_embed (bool): Whether to add time embeddings.
    """
    def __init__(self, 
                 in_channels: int = 9+4, # bands + delta t + DOY + seasonal NDVI + z-score NDVI 
                 latent_dim: int = 64,
                 use_time_embed: bool = True):
        super().__init__()
        self.encoder = VAEEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = VAEDecoder(out_channels=in_channels, latent_dim=latent_dim)
        self.use_time_embed = use_time_embed            # pluggable MLP (multi-layer perceptron)
        if use_time_embed:                              # here. It is flexible, extensible,
            self.time_embed = TimeEmbedding(latent_dim) # able to incorporate event-aware embeddings
                                                        # like monsoon, construction season, days since last rainfall, etc.

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Apply the reparameterization trick to sample z.

        Args:
            mu (Tensor): Latent means.
            logvar (Tensor): Latent log-variances.
        Returns:
            Tensor: Sampled latent tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # random noise
        return mu + eps * std

    def forward(self, x: Tensor, delta_t: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the full VAE.

        Args:
            x (Tensor): Input patch (B, C, 128, 128).
            delta_t (Tensor, optional): Δt embedding (B, 1).
        Returns:
            recon (Tensor): Reconstruction of x.
            mu (Tensor): Latent means.
            logvar (Tensor): Latent log-variances.
        """
        mu, logvar = self.encoder(x)
        # prevent extreme var estimates
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)

        if self.use_time_embed and delta_t is not None:
            time_vec = self.time_embed(delta_t) # (B, latent_dim)
            z = z + time_vec # or torch.cat([z, time_vec], dim=-1) if ok with expanding decoder input. May explode latent space size
                             # It concatenates time embedding to end of z. Gets larger vector with time as its own dimension
                             # But we will need to increase decoders input dimension to match (e.g., latent_dim*2)

        recon = self.decoder(z)
        return recon, mu, logvar

    def compute_loss(self, 
                     x_t0: Tensor, # used indirectly to get mu_t0 before calling this
                     x_t1: Tensor, # used for reconstruction loss
                     recon: Tensor, # model's output for x_t1
                     mu: Tensor, # from encoding x_t1
                     logvar: Tensor, # from encoding X_t0 separately
                     mu_t0: Optional[Tensor] = None,
                     slow_weight: float = 0.1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the VAE loss: reconstruction + KL divergence + optional slow-feature loss.

        Args:
            x_t0 (Tensor): Input at t0 for slow loss.
            x_t1 (Tensor): Ground truth at t1.
            recon (Tensor): Reconstruction of x_t1.
            mu (Tensor): Latent means for x_t1.
            logvar (Tensor): Latent logvars for x_t1.
            mu_t0 (Tensor, optional): Latent means at t0.
            slow_weight (float): Weight for slow-feature penalty.
        Returns:
            total, recon_loss, kl_loss, slow_loss
        """
        # Reconstruction loss (per-pixel MSE)
        recon_loss = F.mse_loss(recon, x_t1, reduction='mean')

        # KL divergence between posterior and unit Gaussian
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        slow_loss = 0.0
        if mu_t0 is not None:
            slow_loss = F.mse_loss(mu_t0, mu)

        total = recon_loss + kl_loss + slow_weight * slow_loss
        return total, recon_loss, kl_loss, slow_loss

# --- fully convolutional versions

# 
# Fully-convolutional architecture will allow for full-size change detection heatmap and mask creation
# 
# This gives:
# •	A 16×16 latent spatial grid, so at inference you can reconstruct or compute Δμ at full 128×128 resolution simply by feeding through decoder—no external upsampling needed.
# •	Skip‐connections ensure fine‐grained detail from early layers flows directly into the decoder.
# •	The SE block learns to emphasize the most informative of your 12 channels (NDVI vs. elevation vs. slope, etc.).
# 

class SEBlock(nn.Module):
    """
    Channel-wise attention (Squeeze-and-Excitation).

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for bottleneck.
    """
    def __init__(self, channels:int, reduction:int = 8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # squeeze to have one number per channel --> (B, C, 1, 1)
            nn.Conv2d(channels, channels//reduction, kernel_size=1),    # bottleneck
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, kernel_size=1),    # re-expand
        nn.Sigmoid(),                   # produce gating weights in (0,1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply channel-wise gating.

        Args:
            x (Tensor): Input feature map (B, C, H, W).
        Returns:
            Tensor: Recalibrated feature map.
        """
        return x * self.gate(x)         # channel-wise rescale

class FCVAEEncoder(nn.Module):
    """
    Fully-convolutional VAE encoder producing spatial μ and logvar maps with skip connections.

    → Input: (B, in_channels, 128,128)
    → f1: (B,  32, 64,64)
    → f2: (B,  64, 32,32)
    → f3: (B, 128, 16,16)
    Emits μ, logvar maps: (B, latent_dim, 16,16)

    Args:
        in_channels (int): Number of input channels.
        latent_dim (int): Number of channels for latent maps.
    """
    def __init__(self, in_channels:int, latent_dim:int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,  32, kernel_size=3, stride=2, padding=1),  # 128→64
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.se1   = SEBlock(32)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,  64, kernel_size=3, stride=2, padding=1),  #  64→32
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  #  32→16
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        # project to spatial μ and logvar maps
        self.conv_mu     = nn.Conv2d(128, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(128, latent_dim, kernel_size=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass producing μ, logvar maps and skip features.

        Args:
            x (Tensor): Input (B, C, 128, 128).
        Returns:
            mu (Tensor): Latent mean map (B, latent_dim, 16,16).
            logvar (Tensor): Latent logvar map (B, latent_dim, 16,16).
            skips (tuple): Skip feature maps for decoder.
        """
        f1 = self.se1(self.conv1(x))
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        mu     = self.conv_mu(f3)
        logvar = self.conv_logvar(f3)
        return mu, logvar, (f1, f2, f3)

class FCVAEDecoder(nn.Module):
    """
    Fully-convolutional VAE decoder with skip concatenation.

    Args:
        out_channels (int): Number of output channels.
        latent_dim (int): Number of latent channels.
    """
    def __init__(self, out_channels:int, latent_dim:int):
        super().__init__()
        # up 16→32
        self.up1    = nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1)
        self.conv_d1 = nn.Sequential(
            nn.BatchNorm2d(128+64), nn.ReLU(inplace=True),
            nn.Conv2d(128+64, 128, kernel_size=3, padding=1),
        )
        # up 32→64
        self.up2    = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_d2 = nn.Sequential(
            nn.BatchNorm2d(64+32), nn.ReLU(inplace=True),
            nn.Conv2d(64+32, 64, kernel_size=3, padding=1),
        )
        # up 64→128
        self.up3    = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv_d3 = nn.Sequential(
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
        )
        # final reconstruction
        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z: Tensor, skips: tuple) -> Tensor:
        """
        Decode latent maps to full resolution with skip connections.

        Args:
            z (Tensor): Latent map (B, latent_dim, 16,16).
            skips (tuple): Feature maps from encoder.
        Returns:
            Tensor: Reconstructed output (B, out_channels,128,128).
        """
        f1, f2, f3 = skips
        x = self.up1(z)                       # → (B,128,32,32)
        x = torch.cat([x, f2], dim=1)         # → (B,192,32,32)
        x = self.conv_d1(x)                   # → (B,128,32,32)

        x = self.up2(x)                       # → (B, 64,64,64)
        x = torch.cat([x, f1], dim=1)         # → (B, 96,64,64)
        x = self.conv_d2(x)                   # → (B, 64,64,64)

        x = self.up3(x)                       # → (B, 32,128,128)
        x = self.conv_d3(x)                   # → (B, 32,128,128)

        return self.final(x)                  # → (B, out_channels,128,128)

class FCVAE(nn.Module):
    """
    Fully-convolutional VAE with spatial latent grid and time embedding support.

    Args:
        in_channels (int): Number of input channels.
        latent_dim (int): Latent dimensionality.
        use_time_embed (bool): Whether to include time embeddings.
    """
    def __init__(self,
                 in_channels: int = 12,     # your 12 inputs
                 latent_dim:   int = 64,
                 use_time_embed: bool = True):
        super().__init__()
        self.encoder      = FCVAEEncoder(in_channels, latent_dim)
        self.decoder      = FCVAEDecoder(out_channels=in_channels, latent_dim=latent_dim)
        self.use_time_embed = use_time_embed
        if use_time_embed:
            self.time_embed = TimeEmbedding(latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Spatial reparameterization trick for µ/logvar maps
        """
        std = (0.5 * logvar).exp()          # (B,LD,16,16)
        eps = torch.randn_like(std)
        return mu + eps * std               # (B,LD,16,16)

    def forward(self, x: Tensor, delta_t: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for FCVAE with optional time addition.

        Args:
            x (Tensor): Input (B, C,128,128).
            delta_t (Tensor, optional): Δt (B,1).
        Returns:
            recon, mu, logvar
        """
        # 1) encode → spatial mu,logvar + skips
        mu, logvar, skips = self.encoder(x)

        # 2) sample z in the *spatial* latent grid
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)

        # 3) optionally add a broadcasted time-embedding
        if self.use_time_embed and delta_t is not None:
            # delta_t: (B,1) → (B,LD) → (B,LD,1,1) → broadcast to (B,LD,16,16)
            tvec = self.time_embed(delta_t)               # (B,LD)
            tvec = tvec.view(tvec.size(0), tvec.size(1), 1, 1)
            z = z + tvec

        # 4) decode with U-Net skips → full res recon
        recon = self.decoder(z, skips)

        return recon, mu, logvar
    
    def compute_loss(self,
                     x_t0: torch.Tensor,
                     x_t1: torch.Tensor,
                     recon: torch.Tensor,
                     mu: torch.Tensor,
                     logvar: torch.Tensor,
                     mu_t0: Optional[torch.Tensor] = None,
                     slow_weight: float = 0.1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute loss for FCVAE: reconstruction + KL + slow-feature.

        Exactly as in VAE:
        - reconstruction MSE
        - KL divergence
        - optional 'slow' term between mu_t0 and mu_t1
        """
        # 1) per‐pixel L2 between recon and x_t1
        recon_loss = F.mse_loss(recon, x_t1, reduction="mean")

        # 2) KL divergence to N(0,1)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 3) slow‐feature penalty
        slow_loss = 0.0
        if mu_t0 is not None:
            slow_loss = F.mse_loss(mu_t0, mu, reduction="mean")

        total = recon_loss + kl_loss + slow_weight * slow_loss
        # 
        # yields:
        # loss, recon_l, kl_l, slow_l = model.compute_loss(
        # x0,      # t₀
        # x1,      # t₁
        # recon1,  # output of forward(x1)
        # mu1,     # mu from forward(x1)
        # logvar1, # logvar from forward(x1)
        # mu0,     # mu from encoder(x0), passed in
        # slow_weight=args.slow_weight
        # )   
        return total, recon_loss, kl_loss, slow_loss

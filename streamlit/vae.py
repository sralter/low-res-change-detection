# vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import time
import logging
from typing import Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    def forward(self, x):
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


    def forward(self, z):
        # z: (batch_size, latent_dim)
        x = self.fc(z)                 # (B, 256*8*8)
        x = x.view(-1, 256, 8, 8)      # (B, 256, 8, 8)
        x = self.decoder(x)           # (B, out_channels, 128, 128)
        return x

class TimeEmbedding(nn.Module):
    """
    Encodes delta time (Δt) as a vector of same size as latent_dim
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, delta_t):
        # delta_t: (B, 1)
        return self.mlp(delta_t)

# --- VAE wrapper class

class VAE(nn.Module):
    """
    VAE wrapper that incorporates:
        encoder to
        reparameterization to
        decoder to
        loss function
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

    def reparameterize(self, mu, logvar):
        """Sample z using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # random noise
        return mu + eps * std

    def forward(self, x, delta_t=None):
        """
        x: (B, C, H, W)
        delta_t: (B, 1) - in days
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
                     x_t0,              # used indirectly to get mu_t0 before calling this
                     x_t1,              # used for reconstruction loss
                     recon,             # model's output for x_t1
                     mu,                # from encoding x_t1
                     logvar,            # from encoding X_t0 separately
                     mu_t0=None,
                     slow_weight=0.1):
        """
        VAE loss = reconstruction loss + KL divergence + slow feature loss
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
    """Channel-wise attention (Squeeze-and-Excitation)."""
    def __init__(self, channels:int, reduction:int=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.gate(x)

class FCVAEEncoder(nn.Module):
    """
    → Input: (B, in_channels, 128,128)
    → f1: (B,  32, 64,64)
    → f2: (B,  64, 32,32)
    → f3: (B, 128, 16,16)
    Emits μ, logvar maps: (B, latent_dim, 16,16)
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
        self.conv_mu     = nn.Conv2d(128, latent_dim,   kernel_size=1)
        self.conv_logvar = nn.Conv2d(128, latent_dim,   kernel_size=1)

    def forward(self, x):
        f1 = self.se1(self.conv1(x))
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        mu     = self.conv_mu(f3)
        logvar = self.conv_logvar(f3)
        return mu, logvar, (f1, f2, f3)


class FCVAEDecoder(nn.Module):
    """
    Takes z of shape (B, latent_dim, 16,16)
    + skips=(f1,f2,f3):
      f2:(B,64,32,32) → cat after up1
      f1:(B,32,64,64) → cat after up2
    Returns recon: (B, out_channels,128,128)
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

    def forward(self, z, skips):
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
        """ Reparam trick: mu/logvar are (B,LD,16,16) now """
        std = (0.5 * logvar).exp()          # (B,LD,16,16)
        eps = torch.randn_like(std)
        return mu + eps * std               # (B,LD,16,16)

    def forward(self, x: Tensor, delta_t: Tensor=None):
        """
        x:        (B, C,128,128)
        delta_t:  (B,1) in days
        returns:  recon (B,C,128,128), mu (B,LD,16,16), logvar (B,LD,16,16)
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
                     slow_weight: float = 0.1):
        """
        exactly as in your old VAE:
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

# # in training, when batching from the dataset, we'll extract delta_t like this:
# batch = dataset[index] # shape (2, C, H, W)
# x = batch[1] # t1 image
# delta_t_val = batch[1][-4] # index of delta t in the channels
# delta_t = delta_t_val[0, 0].view[1, 1] # (B, 1)
# or extract delta_t cleanly in the dataset return format like:
# return torch.from_numpy(pair).float(), torch.tensor([dt_days], dtype=torch.float32)
# then training loop becomes:
# for x, delta_t in dataloader:
#   recon, mu, logvar = model(x, delta_t=delta_t)
# 
# Let’s say you’ve loaded a batch from the dataset as:
# # tile: shape (2, C, H, W)
# x_t0 = batch[:, 0]  # (B, C, H, W)
# x_t1 = batch[:, 1]  # (B, C, H, W)
# 
# Then in training:
# mu_t0, _ = model.encoder(x_t0)  # just get mean from t0
# recon, mu_t1, logvar_t1 = model(x_t1, delta_t=delta_t)
# loss, recon_loss, kl_loss, slow_loss = model.compute_loss(
#     x_t0=x_t0,
#     x_t1=x_t1,
#     recon=recon,
#     mu=mu_t1,
#     logvar=logvar_t1,
#     mu_t0=mu_t0,
#     slow_weight=0.1
# )

# # --- Vizualization of original and reconstruction

# def stretch(img, p_low=1, p_high=99):
#     """Percentile stretch function."""
#     lo, hi = np.percentile(img, (p_low, p_high))
#     return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

# def to_rgb(arr, r_i, g_i, b_i):
#     """Build an (H, W, 3) uint8-like array from a (C, H, W) tensor and channel indices."""
#     rgb = np.moveaxis(arr[[r_i, g_i, b_i]].numpy(), 0, -1)
#     return stretch(rgb)

# def normalize_patch(tensor):
#     """
#     Given a (C, H, W) torch tensor, scale it to [0,1]
#     using its own min/max.
#     """
#     t = tensor.clone()
#     tmin, tmax = t.min(), t.max()
#     return (t - tmin) / (tmax - tmin + 1e-6)

# def inspect_reconstruction(dataset, index, encoder, decoder, save_path: Path = None):
#     """
#     Show the original vs reconstructed RGB patch using a trained or initialized VAE.

#     Parameters:
#     - dataset: instance of ZarrChangePairDataset
#     - index: sample index to inspect
#     - encoder: instance of VAEEncoder
#     - decoder: instance of VAEDecoder
#     - save_path: optional Path to save PNG and .npy version
#     """

#     # 1) Extract original image (C, H, W) — pick t1 for now
#     tile = dataset[index]            # (2, C, H, W)
#     raw    = tile[0]                 # (C, H, W)
#     normed = normalize_patch(raw)    # (C, H, W) in [0,1]
#     t1_img = normed.unsqueeze(0)     # (1, C, H, W)

#     # set 'coders to eval (future: and load checkpoint)
#     encoder.eval()
#     decoder.eval()
#     # state = torch.load("vae_checkpoint.pt", map_location="cpu")
#     # encoder.load_state_dict(state["encoder"])
#     # decoder.load_state_dict(state["decoder"])
#     # encoder.eval()
#     # decoder.eval()

#     # 2) Forward pass through VAE
#     with torch.no_grad():
#         mu, logvar = encoder(t1_img)
#         z = mu  # or sample with reparameterization if needed
#         recon = decoder(z)  # shape: (1, C, H, W)

#     # 3) Get RGB indices
#     try:
#         r_i = dataset.bands.index('r')
#         g_i = dataset.bands.index('g')
#         b_i = dataset.bands.index('b')
#     except ValueError:
#         raise RuntimeError(f"Need bands ['r','g','b'] in dataset.bands, got {dataset.bands!r}")

#     # use true sigmoid output rather than over-stretched output
#     recon_img = recon[0].cpu().numpy()  # shape (C, H, W)
#     rgb_recon = np.moveaxis(recon_img[[r_i,g_i,b_i]], 0, -1)
#     rgb_recon = np.clip(rgb_recon, 0, 1)

#     # 4) Format images for display
#     orig_np    = t1_img[0].cpu().numpy()      # shape (C,H,W)
#     rgb_orig   = np.moveaxis(orig_np[[r_i,g_i,b_i]],0,-1)
#     rgb_orig   = np.clip(rgb_orig, 0, 1)
#     rgb_recon = to_rgb(recon[0], r_i, g_i, b_i)

#     # sanity check - if range like [1.0, 1.0], this means flattening has occurred
#     print("orig R-band min/max:", rgb_orig[...,0].min(), rgb_orig[...,0].max())
#     print("recon R-band min/max:", rgb_recon[...,0].min(), rgb_recon[...,0].max())

#     # 5) Plot
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
#     fig.suptitle(f"Reconstruction of Index {index}", fontsize=18)
#     for ax, img, title in zip(axes, [rgb_orig, rgb_recon], ["Original", "Reconstructed"]):
#         ax.imshow(img)
#         ax.get_xaxis().set_ticks([])
#         ax.get_yaxis().set_ticks([])
#         ax.set_title(title, fontsize=15)

#     # 6) Save if requested
#     if save_path:
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         png_fp = save_path.with_name(f"{save_path.stem}_{index}.png")
#         npy_fp = save_path.with_name(f"{save_path.stem}_{index}.npy")

#         fig.savefig(png_fp, bbox_inches='tight')
#         np.save(npy_fp, recon[0].cpu().numpy())

#     plt.show()
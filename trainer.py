"""
trainer.py

Training loop and utilities for Variational Autoencoder (VAE) models on change-detection datasets.

Contains:
  - VAETrainer: orchestrates training, validation, checkpointing, logging, and hard-negative sampling.

Supports:
  - Weighted sampling for hard-negative mining
  - TensorBoard logging
  - Early stopping
  - Learning rate scheduling
  - Sample-weight updates based on reconstruction error
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
import numpy as np
import time
import logging
from typing import Optional
from tqdm import tqdm

class VAETrainer:
    """
    Trainer for VAE models on change-detection datasets.

    Handles:
      - Device setup (CPU/GPU/MPS)
      - Data loading with weighted sampling for hard-negative mining
      - TensorBoard logging
      - Training and validation loops with slow-feature regularization
      - Learning rate scheduling and checkpointing
      - Sample-weight updates based on per-sample reconstruction error
    """
    def __init__(self,
                 model: nn.Module,
                 train_dataset,
                 val_dataset,
                 out_dir: str,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 scheduler_step: int = 10,
                 scheduler_gamma: float = 0.5,
                 hard_power: float = 1.0,
                 device: Optional[str] = None):
        """
        Initialize the VAETrainer.

        Args:
            model: VAE model instance to train.
            train_dataset: Dataset for training.
            val_dataset: Dataset for validation.
            out_dir: Directory to save checkpoints and logs.
            batch_size: Samples per batch.
            num_workers: DataLoader worker count.
            lr: Learning rate.
            weight_decay: Weight decay for optimizer.
            scheduler_step: Epoch interval for LR scheduler step.
            scheduler_gamma: LR decay factor.
            hard_power: Exponent for sample-weight updates.
            device: Device identifier (e.g., 'cpu', 'cuda', 'mps').
        """

        # Setup device
        if device:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        # Output paths
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.out_dir / "best_model.pt"
        self.log_path = self.out_dir / "training_log.txt"

        # Setup logger
        self.logger = self._setup_logger()

        # Setup TensorBoard
        self.writer = SummaryWriter(
            log_dir=str(self.out_dir / "tensorboard"),
            flush_secs=5)

        # Load model and send to device
        self.model = model.to(self.device)

        # Create dataloaders
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        # Keep track of a per‐sample “difficulty” for hard‐negative mining.
        N = len(train_dataset)
        # start uniform (all weights = 1.0)
        self.sample_weights = np.ones(N, dtype=np.float32)
        self.hard_power = hard_power
        # Build the first train_loader with uniform sampling
        sampler = WeightedRandomSampler(weights=self.sample_weights, num_samples=N, replacement=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,  # especially if you’re using GPU
            persistent_workers=(self.num_workers > 0)
            # prefetch_factor=2 # optional: number of batches to prefetch per worker
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True)

        # Optimizer & scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        # Tracking
        self.best_val_loss = float("inf")
        self.epoch = 0

        self.logger.info(f"Initialized VAETrainer on device: {self.device}")

    def _setup_logger(self):
        """
        Configure file and console logger.

        Returns:
            logger: Configured logging.Logger instance.
        """
        logger = logging.getLogger("VAETrainer")
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers in Jupyter
        if not logger.handlers:
            fh = logging.FileHandler(self.log_path)
            ch = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger
    
    def train_one_epoch(self, 
                        slow_weight: float = 0.1, 
                        beta: float = 1.0) -> float:
        """
        Run one training epoch.

        Args:
            slow_weight: Weight for slow-feature regularization.
            beta: KL divergence scaling factor.

        Returns:
            avg_loss: Average total loss over the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        slow_loss_sum = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Train]", leave=False)
        
        for pair, delta_t in pbar:
            # pair: (B, 2, C, H, W), delta_t: (B, )
            # time-axis is axis=1, channel-axis is 2
            x_t0 = pair[:, 0].to(self.device) # (B, C, H, W)
            x_t1 = pair[:, 1].to(self.device) # (B, C, H, W)
            delta_t = delta_t.to(self.device).unsqueeze(1)  # (B, 1)

            self.optimizer.zero_grad()

            # Encode t0
            mu_t0, _, _ = self.model.encoder(x_t0)

            # Forward pass on t1
            recon, mu_t1, logvar_t1 = self.model(x_t1, delta_t=delta_t)

            # Loss
            loss, recon_loss, kl_loss, slow_loss = self.model.compute_loss(
                x_t0=x_t0,
                x_t1=x_t1,
                recon=recon,
                mu=mu_t1, 
                logvar=logvar_t1, 
                mu_t0=mu_t0, 
                slow_weight=slow_weight
            )

            # re-scale KL term
            loss = recon_loss + beta * kl_loss + slow_weight * slow_loss

            loss.backward()
            # clip any crazy gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Logging
            epoch_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            slow_loss_sum += slow_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}",
                "slow": f"{slow_loss.item():.4f}",
            })

        self.scheduler.step()

        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = epoch_loss / num_batches
        avg_recon = recon_loss_sum / num_batches
        avg_kl = kl_loss_sum / num_batches
        avg_slow = slow_loss_sum / num_batches

        self.logger.info(
            f"[Train] Epoch {self.epoch+1} - Total: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Slow: {avg_slow:.4f}"
        )

        self.writer.add_scalar("Loss/train_total", avg_loss, self.epoch)
        self.writer.add_scalar("Loss/train_recon", avg_recon, self.epoch)
        self.writer.add_scalar("Loss/train_kl", avg_kl, self.epoch)
        self.writer.add_scalar("Loss/train_slow", avg_slow, self.epoch)

        self.writer.flush()

        return avg_loss

    def _save_checkpoint(self):
        """
        Save model and optimizer state to disk as the best checkpoint.
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "val_loss": self.best_val_loss
        }, self.checkpoint_path)

    @torch.no_grad() # saves memory and speeds up function
    def validate_one_epoch(self, 
                           slow_weight: float = 0.1, 
                           beta: float = 1.0) -> float:
        """
        Run one validation epoch without gradient updates.

        Args:
            slow_weight: Weight for slow-feature regularization.
            beta: KL divergence scaling factor.

        Returns:
            avg_loss: Average validation loss.
        """
        self.model.eval()

        val_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        slow_loss_sum = 0.0
        recon_errors = []
        latent_deltas = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Val]", leave=False)

        for pair, delta_t in pbar:
            # pair, delta_t = batch
            x_t0 = pair[:, 0].to(self.device)
            x_t1 = pair[:, 1].to(self.device)
            delta_t = delta_t.to(self.device).unsqueeze(1)

            mu_t0, _, _ = self.model.encoder(x_t0)
            recon, mu_t1, logvar_t1 = self.model(x_t1, delta_t=delta_t)

            loss, recon_loss, kl_loss, slow_loss = self.model.compute_loss(
                x_t0=x_t0,
                x_t1=x_t1, 
                recon=recon, 
                mu=mu_t1, 
                logvar=logvar_t1, 
                mu_t0=mu_t0, 
                slow_weight=slow_weight
            )

            # re-scale KL term
            loss = recon_loss + beta * kl_loss + slow_weight * slow_loss

            val_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            slow_loss_sum += slow_loss.item()

            # Per-patch metrics
            recon_errors.append(F.l1_loss(recon, x_t1, reduction='none').mean(dim=(1,2,3)).cpu().numpy())
            latent_deltas.append(torch.norm(mu_t1 - mu_t0, p=2, dim=1).cpu().numpy())

            pbar.set_postfix({
                "val_loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}",
                "slow": f"{slow_loss.item():.4f}",
            })

        # Aggregate
        num_batches = len(self.val_loader)
        avg_loss = val_loss / num_batches
        avg_recon = recon_loss_sum / num_batches
        avg_kl = kl_loss_sum / num_batches
        avg_slow = slow_loss_sum / num_batches

        # Flatten per-image metrics
        recon_all = np.concatenate(recon_errors)
        latent_all = np.concatenate(latent_deltas)

        self.logger.info(
            f"[Val] Epoch {self.epoch+1} - Total: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Slow: {avg_slow:.4f}"
        )
        self.logger.info(
            f"[Val-Metrics] Recon Mean: {recon_all.mean():.4f} ± {recon_all.std():.4f} | Latent Δ: {latent_all.mean():.4f} ± {latent_all.std():.4f}"
        )

        # Save best checkpoint
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint()
            self.logger.info(f"New best model saved (val_loss={avg_loss:.4f})")

        # Log side-by-side images for the first batch, every 5 epochs
        if self.epoch % 5 == 0:  # log every 5 epochs
            try:
                # pull off channel-first (3,H,W)
                x_vis = x_t1[0].detach().cpu()
                recon_vis = recon[0].detach().cpu()

                # “dataset_root” is whichever dataset actually has .bands (e.g. a ZarrChangePairDataset)
                if isinstance(self.train_loader.dataset, torch.utils.data.ConcatDataset):
                    dataset_root = self.train_loader.dataset.datasets[0]
                else:
                    dataset_root = self.train_loader.dataset
                r_i = dataset_root.bands.index("R")
                g_i = dataset_root.bands.index("G")
                b_i = dataset_root.bands.index("B")

                def to_rgb(tensor):
                    # returns (3, H, W), exactly what add_image(dataformats='CHW') wants
                    rgb = tensor[[r_i, g_i, b_i]]
                    return (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

                self.writer.add_image("RGB/t1_orig", to_rgb(x_vis), self.epoch)
                self.writer.add_image("RGB/reconstruction",
                                      to_rgb(recon_vis),
                                      self.epoch)
            except Exception as e:
                self.logger.warning(f"Could not log images to TensorBoard: {e}")

        self.writer.add_scalar("Loss/val_total", avg_loss, self.epoch)
        self.writer.add_scalar("Loss/val_recon", avg_recon, self.epoch)
        self.writer.add_scalar("Loss/val_kl", avg_kl, self.epoch)
        self.writer.add_scalar("Loss/val_slow", avg_slow, self.epoch)

        self.writer.add_scalar("Metrics/recon_mean", recon_all.mean(), self.epoch)
        self.writer.add_scalar("Metrics/recon_std",  recon_all.std(), self.epoch)
        self.writer.add_scalar("Metrics/latent_mean", latent_all.mean(), self.epoch)
        self.writer.add_scalar("Metrics/latent_std",  latent_all.std(), self.epoch)

        return avg_loss

    def _rebuild_train_loader(self):
        """
        Rebuild the training DataLoader using updated sample weights for hard-negative mining.
        """
        N = len(self.train_dataset)
        # PyTorch’s WeightedRandomSampler expects a 1‐D list/array of length N
        t0 = time.time()
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(self.sample_weights),
            num_samples=N,
            replacement=True
        )
        t1 = time.time()
        self.logger.info(f"[Sampler Timing] Built WeightedRandomSampler for N={N} in {t1 - t0:.3f}s")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers
        )

    @torch.no_grad()
    def _update_sample_weights(self):
        """
        Compute per-sample reconstruction error and update sample weights accordingly.
        
        Runs over every index in train_dataset just once, computing a per-sample “reconstruction error.”
        Uses that error to set self.sample_weights[ idx ] = (error + ε). 
        You can then choose to normalize or clip so that weights don't collapse to zero.
        """
        self.model.eval()
        N = len(self.train_dataset)
        new_errs = np.zeros(N, dtype=np.float32)

        # We need to compute reconstruction error for each sample index.
        # Because __getitem__ returns (pair, dt), we can iterate i -> (inp, dt).
        for idx in range(N):
            pair, delta_t = self.train_dataset[idx]
            # pair: shape=(2, C, H, W), delta_t: scalar in [0,1]
            x_t0 = pair[0].unsqueeze(0).to(self.device)  # (1, C, H, W)
            x_t1 = pair[1].unsqueeze(0).to(self.device)  # (1, C, H, W)
            # force dt to have shape (1,1):
            if delta_t.dim() == 0:
                dt = delta_t.unsqueeze(0).unsqueeze(1).to(self.device)  # (1,1)
            else:
                dt = delta_t.unsqueeze(0).to(self.device) # if delta_t already 1-d, this yields (1,1)

            # encode t0
            mu_t0, _, _ = self.model.encoder(x_t0)
            # decode/predict t1
            recon, mu_t1, logvar_t1 = self.model(x_t1, delta_t=dt)

            # compute per‐pixel L1 error on each sample (batch size=1)
            per_pixel = F.l1_loss(recon, x_t1, reduction='none')  # shape=(1, C, H, W)
            per_sample = per_pixel.mean(dim=(1,2,3)).item()       # scalar

            # store as new weight (add a tiny epsilon so nothing is zero)
            new_errs[idx] = per_sample + 1e-6

        # raise to hard_power
        new_weights = new_errs ** self.hard_power

        # normalize weights so they sum to 1
        new_weights = new_weights / float(new_weights.sum() + 1e-8)
        self.sample_weights = new_weights

    def train(self, 
            num_epochs: int = 50,
            slow_weight: float = 0.1,
            early_stopping: Optional[int] = None):
        """
        Train the model for N epochs, optionally stopping early if validation doesn't improve.
        
        Usage:
        trainer = VAETrainer(
            model=vae,
            train_dataset=train_set,
            val_dataset=val_set,
            out_dir="results/vae_run1",
            batch_size=16,
            lr=1e-3
        )
        trainer.train(num_epochs=50, slow_weight=0.1, early_stopping=5)

        Args:
            num_epochs (int): total number of epochs
            slow_weight (float): weight for slow feature regularization
            early_stopping (int): number of epochs with no improvement before stopping
        """
        start_time = time.time()
        no_improve = 0

        # how many epochs to anneal over?
        # e.g. 10% of the total, but at least 1 so we don't divide by zero
        ramp_epochs = max(1, int(num_epochs * 0.1))

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.logger.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

            # (re-)build train_loader from current sample_weights
            self._rebuild_train_loader()

            # linear B from 0 to 1 over first 1/10 epochs
            beta = min(1.0, float(epoch+1) / ramp_epochs)

            train_loss = self.train_one_epoch(slow_weight=slow_weight, beta=beta)
            val_loss = self.validate_one_epoch(slow_weight=slow_weight, beta=beta)
            self.logger.info(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                no_improve = 0
            else:
                no_improve += 1

            if early_stopping and no_improve >= early_stopping:
                self.logger.info(f"Early stopping triggered after {no_improve} epochs without improvement.")
                break

            # after each epoch, update sample_weights using latest model
            self._update_sample_weights()
            self.logger.info(f"Initialized VAETrainer on device: {self.device}, hard_power={self.hard_power}")

        duration = time.time() - start_time
        self.logger.info(f"\nTraining completed in {duration/60:.2f} min")
        
        # close TensorBoard writer
        self.writer.close()

# to launch TensorBoard, from terminal run:
# tensorboard --logdir results/vae_run1/tensorboard
# usually the URL is http://localhost:6006

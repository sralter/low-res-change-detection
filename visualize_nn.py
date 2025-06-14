"""
visualize_nn.py

Inspect and visualize the architecture of a trained FCVAE model.

Steps:
  1. Load a saved checkpoint and read the learned latent dimension.
  2. Reconstruct the full FCVAE module with that latent size.
  3. Define an `EncoderMuOnly` wrapper that returns only the μ map.
  4. Use `torchsummary.summary` to print a layer-by-layer summary of the encoder.
  5. Use `torchviz.make_dot` to render the computational graph of the full encoder.
"""

import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot
import logging
import pymaap
pymaap.init_general_logger()

from vae import FCVAE, FCVAEEncoder

class EncoderMuOnly(nn.Module):
    """
    Wrap an existing FCVAEEncoder to expose *only* its μ output.

    This module reuses the same conv1, se1, conv2, conv3, conv_mu layers
    from the full encoder, but discards logvar and skip outputs.

    Args:
      encoder (FCVAEEncoder): the trained encoder to wrap

    Forward:
      x: Tensor of shape (B, in_channels, 128, 128)

    Returns:
      mu: Tensor of shape (B, latent_dim, H_lat, W_lat)
    """
    def __init__(self, encoder: FCVAEEncoder):
        super().__init__()
        # reuse encoder layers
        self.conv1   = encoder.conv1
        self.se1     = encoder.se1
        self.conv2   = encoder.conv2
        self.conv3   = encoder.conv3
        self.conv_mu = encoder.conv_mu

    def forward(self, x):
        # replicate exactly the encoder’s forward pass, but return only mu:
        x = self.conv1(x)       # → (B, 32, 64, 64)
        x = self.se1(x)         # → (B, 32, 64, 64)
        x = self.conv2(x)       # → (B, 64, 32, 32)
        x = self.conv3(x)       # → (B, 128,16,16)
        mu = self.conv_mu(x)    # → (B, trained_latent_dim,16,16)
        return mu

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect and visualize a trained FCVAE model architecture"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the checkpoint file (e.g., best_model.pt)"
    )
    parser.add_argument(
        "--in-channels", type=int, default=13,
        help="Number of input channels used at training"
    )
    parser.add_argument(
        "--use-time-embed", action="store_true",
        help="Whether the model was trained with a time embedding"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print torchsummary of the encoder"
    )
    parser.add_argument(
        "--graph", action="store_true",
        help="Render computational graph (requires torchviz)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Step A: load checkpoint and read the latent_dim from encoder.conv_mu.weight
    ckpt = torch.load(args.model, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Infer latent_dim from conv_mu weight
    latent_key = next(
        (k for k in state_dict if k.endswith("encoder.conv_mu.weight")),
        None
    )
    if latent_key is None:
        raise KeyError("Could not find 'encoder.conv_mu.weight' in checkpoint keys.")
    latent_dim = state_dict[latent_key].shape[0]

    # --- Step B: instantiate full VAE (to get all weights) with the same latent_dim
    full_vae = FCVAE(
        in_channels=args.in_channels,
        latent_dim=latent_dim,
        use_time_embed=args.use_time_embed
    )
    full_vae.load_state_dict(state_dict)
    full_vae.eval()

    logging.info(f"Loaded FCVAE(in_channels={args.in_channels}, latent_dim={latent_dim}, "
                 f"use_time_embed={args.use_time_embed})")
    logging.info(full_vae)

    # --- Step C: build an encoder‐only submodule that returns just mu, never a tuple
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step D: call torchsummary on the encoder-only model
    # torchsummary will now see a single‐Tensor output (mu), so no tuple errors.
    if args.summary:
        # Wrap encoder for summary
        encoder_only = EncoderMuOnly(full_vae.encoder).to(device)
        summary(encoder_only, input_size=(args.in_channels, 128, 128), device=str(device))

    if args.graph:
        # Generate computational graph
        dummy_input = torch.randn(1, args.in_channels, 128, 128).to(device)
        dummy_dt = torch.tensor([[1.0]], device=device)
        _, mu, _ = full_vae(dummy_input, delta_t=dummy_dt)

        dot = make_dot(mu, params=dict(full_vae.named_parameters()))
        dot.format = "pdf"
        dot.directory = "model_graphs"
        out_name = f"vae_encoder_graph_{latent_dim}"
        dot.render(out_name)
        logging.info(f"Graph saved to model_graphs/{out_name}.pdf")

if __name__ == "__main__":
    main()

# Usage
# python visualize_nn.py \
#   --model path/to/best_model.pt \
#   --in-channels 13 \
#   --use-time-embed \
#   --summary \
#   --graph

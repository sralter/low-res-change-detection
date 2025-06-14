"""
train_vae.py

Orchestrates end-to-end training and evaluation of a fully-convolutional 
    VAE for geospatial change detection.

Workflow:
  1. Parse CLI args (geohashes, S3 locations, hyperparams, etc.).
  2. Optionally stage Zarr stores locally or stream from S3.
  3. Build train/val/test `ZarrChangePairDataset` splits with global min/max 
        normalization and temporal sampling.
  4. Instantiate `VAETrainer` and run training + validation.
  5. Optionally evaluate on held-out test splits.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import random
import argparse
from torch.utils.data import DataLoader, ConcatDataset
import json
from tempfile import TemporaryDirectory
import s3fs
import logging
from pymaap import init_general_logger
init_general_logger()
from prepare_dataset import open_zarr

from prepare_dataset import ZarrChangePairDataset, compute_global_band_ranges
from vae import FCVAE as VAE # using fully-convolutional version now
from trainer import VAETrainer

def seed_all(seed: int):
    """
    Set the random seed across torch, numpy, random (and all CUDA devices, if available)
    for full reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stage_zarr_locally(
    zarr_uri: str,
    aws_creds: dict | None = None,
    parent_dir: str | None = None
) -> str:
    """
    Download a Zarr store (given its S3 URI) into a local directory and return that path.
    If `parent_dir` is provided, stage under `<parent_dir>/<geohash>_zarr`;
    otherwise use a TemporaryDirectory (auto-cleaned on exit).
    """
    # strip off 's3://'
    bucket_key = zarr_uri.removeprefix("s3://")
    geohash_name = Path(bucket_key).name  # e.g. "9vgm0_zarr"

    if parent_dir is not None:
        # Use the provided parent directory as the staging root.
        local_root = Path(parent_dir)
        local_root.mkdir(parents=True, exist_ok=True)
        local_store = local_root / geohash_name
        local_store.parent.mkdir(parents=True, exist_ok=True)
        # If the folder already exists (from a previous trial), skip re-download:
        if local_store.exists():
            logging.info(f"Reusing existing staged Zarr at {local_store}")
            return str(local_store)
        # Otherwise, proceed to download into local_store
    else:
        # No parent_dir given → fall back to an auto-cleaned TemporaryDirectory
        tmp = TemporaryDirectory(prefix="zarr_stage_")
        local_root = Path(tmp.name)
        local_store = local_root / geohash_name
        local_store.parent.mkdir(parents=True, exist_ok=True)
        # Keep track of tmp so it isn’t garbage-collected immediately
        setattr(stage_zarr_locally, "_all_tmps", 
                getattr(stage_zarr_locally, "_all_tmps", []) + [tmp])

    # pick the right s3fs filesystem
    if aws_creds:
        fs = s3fs.S3FileSystem(
            anon=False,
            key=aws_creds["access_key"],
            secret=aws_creds["secret_access_key"],
        )
    else:
        fs = s3fs.S3FileSystem(anon=True)

    logging.info(f"Downloading {zarr_uri} → {local_store}")
    fs.get(bucket_key, str(local_store), recursive=True)
    print(f"Staged {zarr_uri} → {local_store}")
    return str(local_store)

# Keep track of TemporaryDirectories so they aren’t garbage‐collected immediately
stage_zarr_locally._all_tmps: list[TemporaryDirectory] = []  # type: ignore[attr-defined]

def _make_concat_dataset(
    geohash_list: list[str],
    local_paths: dict[str,str],
    args,
    aws_creds: dict,
    global_min_vals: np.ndarray,
    global_max_vals: np.ndarray#,
    # time_indices: list[tuple[int,int]]
) -> ConcatDataset | ZarrChangePairDataset:
    """
    Construct one or more `ZarrChangePairDataset` objects—one per geohash—
    sampling temporal pairs according to `args.max_step`, `args.time_gap_exp`, and `args.num_time_pairs`.
    If multiple geohashes, wrap them in a `ConcatDataset`, else return the single dataset.
    """
    raw_bands = ["R","G","B","NDVI","elevation","slope","aspect","aspect_mask"]
    stores = []

    for gh in geohash_list:
        zroot = local_paths[gh]
        clim  = f"s3://{args.bucket}/{args.folder}/{gh}/{gh}_climatology"

        # ——— Step A: Open this geohash’s Zarr *just to measure T_this* ———
        ds_temp = open_zarr(zroot, consolidated=True, aws_creds=aws_creds)
        time_dim = next(d for d in ds_temp.dims if "time" in d.lower() or "date" in d.lower())
        T_this = ds_temp.sizes[time_dim]
        ds_temp.close()

        # ——— Step B: Build all valid (i, j) with k in [7..max_step], j < T_this ———
        all_pairs = []
        for i in range(T_this):
            for k in range(7, args.max_step + 1, 7): # limit to week steps
                j = i + k
                if j < T_this:
                    all_pairs.append((i, j))
        if len(all_pairs) == 0:
            raise RuntimeError(f"No valid (i,i+k) pairs for geohash={gh} with max_step={args.max_step} and T_this={T_this}")

        # ——— Step C: Compute weights ∝ (j−i)^time_gap_exp, then sample a fraction ———
        a_ = args.time_gap_exp
        weights = np.array([float((j - i) ** a_) for (i, j) in all_pairs], dtype=np.float64)
        weights /= weights.sum()

        frac = args.num_time_pairs
        if frac <= 0.0:
            raise ValueError(f"--num-time-pairs must be >0.0 (got {frac})")
        if frac >= 1.0 or frac == 1.0:
            chosen_pairs = all_pairs
        else:
            N_draw = max(1, int(np.floor(len(all_pairs) * frac)))
            chosen_idx = np.random.choice(len(all_pairs), size=N_draw, replace=False, p=weights)
            chosen_pairs = [ all_pairs[i] for i in chosen_idx ]

        # ——— Step D: Instantiate this geohash’s dataset with its own chosen_pairs ———
        ds = ZarrChangePairDataset(
            zarr_path           = zroot,
            patch_size          = args.patch_size,
            stride              = args.patch_stride or args.patch_size,
            bands               = raw_bands,
            seasonal_ndvi_path  = f"{clim}/seasonal_ndvi.nc",
            monthly_mu_path     = f"{clim}/monthly_ndvi_mean.nc",
            monthly_sigma_path  = f"{clim}/monthly_ndvi_std.nc",
            aws_creds           = aws_creds,
            stage               = False,
            global_min_vals     = global_min_vals,
            global_max_vals     = global_max_vals,
            time_indices        = chosen_pairs
        )
        stores.append(ds)

    if len(stores) > 1:
        return ConcatDataset(stores)
    else:
        return stores[0]

def build_datasets(
    args,
    aws_creds: dict,
    local_paths: dict[str,str]
) -> tuple[
    ConcatDataset | ZarrChangePairDataset,
    ConcatDataset | ZarrChangePairDataset,
    ConcatDataset | ZarrChangePairDataset | None,
    int
]:
    """
    1) Compute global per-band min/max across train+val+test geohashes.
    2) Build NumPy arrays of those values for normalization.
    3) Call `_make_concat_dataset` to assemble train, val, (optional) test splits.
    4) Return `(train_ds, val_ds, test_ds, in_channels)` for model construction.
    """
    all_ghs = (
        args.train_geohashes
        + args.val_geohashes
        + (args.test_geohashes if args.test_geohashes else [])
    )

    # Suppose bands = ["R","G","B","NDVI","elevation","slope","aspect","aspect_mask"]
    bands = ["R","G","B","NDVI","elevation","slope","aspect","aspect_mask"]

    # 1) Build two dicts mapping band→min and band→max across ALL geohashes:
    global_min_dict, global_max_dict = compute_global_band_ranges(
        bucket    = args.bucket,
        folder    = args.folder,
        geohashes = all_ghs,
        aws_creds = aws_creds
    )

    # 2) Convert them into two NumPy arrays in the same band order:
    min_list = [global_min_dict[b] for b in bands]
    max_list = [global_max_dict[b] for b in bands]
    global_min_vals = np.array(min_list, dtype=np.float32)  # shape=(8,)
    global_max_vals = np.array(max_list, dtype=np.float32)  # shape=(8,)

    # 3) Build each split’s dataset by calling our new `_make_concat_dataset(...)`
    train_ds = _make_concat_dataset(
        geohash_list    = args.train_geohashes,
        local_paths     = local_paths,
        args            = args,
        aws_creds       = aws_creds,
        global_min_vals = global_min_vals,
        global_max_vals = global_max_vals
    )

    val_ds = _make_concat_dataset(
        geohash_list    = args.val_geohashes,
        local_paths     = local_paths,
        args            = args,
        aws_creds       = aws_creds,
        global_min_vals = global_min_vals,
        global_max_vals = global_max_vals
    )

    test_ds = None
    if args.test_geohashes:
        test_ds = _make_concat_dataset(
            geohash_list    = args.test_geohashes,
            local_paths     = local_paths,
            args            = args,
            aws_creds       = aws_creds,
            global_min_vals = global_min_vals,
            global_max_vals = global_max_vals
        )

    in_channels = (
        len(bands) + 1 # 8 original bands, but “aspect” is replaced by sin+cos → +1 channel
        ) + 4 # add 4 metadata channels: (Δt, DOY, seasonal NDVI, z_NDVI)

    return train_ds, val_ds, test_ds, in_channels

def train_and_evaluate(args) -> tuple[float, dict[str,str]]:
    """
    1) Load AWS creds and seed RNGs.
    2) Stage or point at each geohash’s Zarr store.
    3) Build datasets and instantiate `VAETrainer`.
    4) Run `trainer.train(...)` then one final `trainer.validate_one_epoch(...)`.
    5) Return `(best_val_loss, local_paths)` mapping geohash→local_or_s3_path.
    """
    # load creds
    logging.info("Parsing AWS credentials")
    aws_creds = json.load(open(args.aws_creds))

    # reproducibility
    logging.info(f"Setting seed to {args.seed}")
    seed_all(args.seed)

    # --- stage or point at S3
    # if the user wants us to stage everything locally first:
    all_uris = args.train_geohashes + args.val_geohashes + args.test_geohashes
    logging.info(f"User opted to stage zarrs locally? stage_zarr={args.stage_zarr}, stage_zarr_root={args.stage_zarr_root}")
    if args.stage_zarr_root is not None:
        # 1) If the user provided a --stage-zarr-root, use that folder for all geohashes
        local_paths = {
            gh: str(Path(args.stage_zarr_root) / f"{gh}_zarr")
            for gh in all_uris
        }
    elif args.stage_zarr:
        # 2) Otherwise, if they explicitly asked “--stage-zarr” (but no root), download now:
        local_paths = {
            gh: stage_zarr_locally(
                    f"s3://{args.bucket}/{args.folder}/{gh}/{gh}_zarr",
                    aws_creds=aws_creds
                 )
            for gh in all_uris
        }
    else:
        # 3) If neither root nor --stage-zarr are set, stream directly from S3
        local_paths = {
            gh: f"s3://{args.bucket}/{args.folder}/{gh}/{gh}_zarr"
            for gh in all_uris
        }

    logging.info("Instantiating the train and val splits...")
    train_ds, val_ds, _, in_ch = build_datasets(args, aws_creds, local_paths)

    logging.info("Defining trainer")
    model = VAE(in_channels=in_ch, latent_dim=args.latent_dim)
    trainer = VAETrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        out_dir=args.out,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        device=None,
        hard_power=args.hard_power
    )
    # if hard_power == 1.0, treat as uniform sampling (no mining)
    if args.hard_power == 1.0:
        # override sample_weights to pick uniformly (effectively shuffle=True)
        trainer.sample_weights = np.ones(len(train_ds), dtype=np.float32)
    logging.info("Training...")
    trainer.train(num_epochs=args.epochs,
                  slow_weight=args.slow_weight,
                  early_stopping=args.early_stopping)
    
    logging.info("Validation...")
    val_loss = trainer.validate_one_epoch(
        slow_weight=args.slow_weight,
        beta=1.0
    )
    # return the val loss *and* the map of {geohash → staged‐or‐S3‐path}
    return val_loss, local_paths

def evaluate_on_test(args, all_local_paths: dict[str,str]) -> float:
    """
    Load the best checkpoint from `args.out` and rebuild the test split
    (using the same global min/max). Run validation on test data and
    log per-patch L1 reconstruction error and latent Δμ metrics.
    Returns the final `test_loss`.
    """
    # assumes best_model.pt was written into args.out by VAETrainer
    ckpt = torch.load(Path(args.out)/"best_model.pt", map_location="cpu")
    aws_creds = json.load(open(args.aws_creds))
    # only take the test set from the already‐staged paths
    test_ghs = args.test_geohashes
    local_paths = { gh: all_local_paths[gh] for gh in test_ghs }
    _, _, test_ds, in_ch = build_datasets(args, aws_creds, local_paths)
    model   = VAE(in_channels=in_ch, latent_dim=args.latent_dim)
    sd      = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    trainer = VAETrainer(
        model        = model,
        train_dataset= test_ds,   # dummy
        val_dataset  = test_ds,
        out_dir      = args.out,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        scheduler_step  = args.scheduler_step,
        scheduler_gamma = args.scheduler_gamma,
        device       = None,
        hard_power   = args.hard_power
    )
    trainer.val_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)
    test_loss = trainer.validate_one_epoch(
        slow_weight=args.slow_weight, 
        beta=1.0)

    recon_errors, latent_deltas = [], []
    with torch.no_grad():
        for pair, dt in trainer.val_loader:
            x0 = pair[:,0].to(trainer.device)
            x1 = pair[:,1].to(trainer.device)
            dt = dt.to(trainer.device).unsqueeze(1)

            mu0, _ = trainer.model.encoder(x0)
            recon, mu1, _ = trainer.model(x1, delta_t=dt)

            recon_l1 = F.l1_loss(recon, x1, reduction='none') \
                            .mean(dim=(1,2,3)).cpu().numpy()
            latent_norm = torch.norm(mu1 - mu0, dim=1).cpu().numpy()

            recon_errors.append(recon_l1)
            latent_deltas.append(latent_norm)

    recon_all  = np.concatenate(recon_errors)
    latent_all = np.concatenate(latent_deltas)
    logging.info(f"Recon L1 error: {recon_all.mean():.4f} ± {recon_all.std():.4f}")
    logging.info(f"Latent Δμ   : {latent_all.mean():.4f} ± {latent_all.std():.4f}")

    return test_loss

def parse_args(cmd: list[str] | None = None):
    """
    Parse command-line arguments for:
      - data splits (train/val/test geohashes)
      - S3 bucket/folder or staging options
      - model hyperparameters (latent_dim, patch_size, etc.)
      - training settings (epochs, lr, scheduler, early-stopping, etc.)
    Returns an `argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Train VAE for geospatial change detection"
    )
    parser.add_argument("--seed", "-s", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument("--bucket", '-b', type=str, required=True,
        help="Name of S3 bucket, e.g., 'my-bucket-name'"
    )
    parser.add_argument("--folder", "-f", type=str, required=False, default="data",
        help="The folder where all the Zarrs are located in the S3 bucket"
    )
    parser.add_argument("--train-geohashes", '-tr', type=lambda s: s.split(","), required=True,
        help="List of 5-digit geohash codes to train on, e.g., 9vgm0,9vgm1,9vgm2"
        )
    parser.add_argument("--val-geohashes", "-va", type=lambda s: s.split(","), default=[], 
        help=("List of 5-digit geohash codes to validate on"
    ))
    parser.add_argument("--test-geohashes", "-te", type=lambda s: s.split(","), default=[],
        help="(Optional) List of 5-digit geohash codes for test, e.g., 9vgm3,9vgm4"
    )
    parser.add_argument("--out", "-o", type=str, required=True,
        help="Directory for outputs and checkpoints"
    )
    parser.add_argument("--latent-dim", "-l", type=int, default=64,
        help="Dimensionality of the VAE latent space"
    )
    parser.add_argument("--aws-creds", '-A',
        help="Path to JSON file with AWS credentials {'access_key':..., 'secret_access_key':...}"
    )
    parser.add_argument("--patch-size",   type=int, default=128,
        help="Size (in pixels) of each square patch"
    )
    parser.add_argument("--patch-stride", type=int, default=None,
        help="Stride (in pixels) between patch windows; if unset, equals patch-size"
    )
    parser.add_argument("--num-workers",  type=int, default=4,
        help="Number of DataLoader worker processes"
    )
    parser.add_argument("--stage-zarr-root", type=str, default=None, help=(
        "Path to a directory containing pre-downloaded zarr stores (one per geohash). Overrides --stage-zarr."
	))
    parser.add_argument(
        "--stage-zarr", action="store_true", dest="stage_zarr",
        help="If set, download each Zarr locally before training."
    )
    parser.add_argument(
        "--no-stage-zarr", action="store_false", dest="stage_zarr",
        help="Do *not* download Zarrs locally; stream from S3."
    )
    parser.add_argument("--hard-power", type=float, default=1.0, help=(
        "Exponent for hard-negative mining." 
        "If --hard-power = 1.0 (the default), sampling is uniform."
        "Any value > 1.0 will reweight patches by (recon-error ** hard_power)."))
    parser.add_argument("--max-step", type=int, default=1, help=(
        "Maximum temporal 'gap' (in number of frames) to sample. "
        "If --max-step=1, only adjacent (i,i+1) pairs are used. "
        "If >1, all (i, i+k) with 1≤k≤max_step are included."))
    parser.add_argument("--time-gap-exp", type=float, default=1.0, help=(
        "Exponent reflecting how much to weigh the larger time-gapped (k) samples in (i, i+k)."
    ))
    parser.add_argument("--num-time-pairs", type=float, default=0.05, help=(
        "Fraction of all valid (i, i+k) pairs (per geohash) to sample each epoch. "
        "Use small fractions (e.g. 0.05) to avoid huge numbers of patches."
    ))
    parser.set_defaults(stage_zarr=False)
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch-size", "-bs", type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-5)
    parser.add_argument("--scheduler-step",type=int,   default=10)
    parser.add_argument("--scheduler-gamma",type=float,default=0.5)
    parser.add_argument("--slow-weight",   type=float, default=0.1)
    parser.add_argument("--early-stopping", "-e", type=int,   default=5)

     # if cmd is None, parse from sys.argv; otherwise parse from that list
    args = parser.parse_args(cmd)
    return args

def main():
    """
    Entry point:
      - Parse args
      - Call `train_and_evaluate`
      - Log validation loss
      - If `--test-geohashes` provided, call `evaluate_on_test`
    """
    args = parse_args()

    logging.info("Starting training and validation...")
    val_loss, local_paths = train_and_evaluate(args)
    logging.info(f"\n=== Validation loss: {val_loss:.4f} ===")

    # If the user specified --test-geohashes, run the held-out test evaluation:
    if args.test_geohashes:
        logging.info("\nHeld-out test evaluation...")
        test_loss = evaluate_on_test(args, local_paths)
        logging.info(f"=== Test loss: {test_loss:.4f} ===")

if __name__ == "__main__":
    main()

# usage:
# Train on geohashes 9vgm6 and 9vgm7, test on 9vgm8, pulling all data from
# s3://my-bucket/data/{geohash}/{geohash}_zarr and writing outputs to ./results/vae_run1
# python train_vae.py \
#   -s 42 \                          # random seed
#   -b my-bucket \                   # S3 bucket name
#   -f data \                        # prefix inside bucket e.g. my-bucket/data
#   -tr 9vgm6+9vgm7 \                # geohashes for train+val
#   -te 9vgm8 \                       # geohashes for held-out test
#   -o results/vae_run1 \            # output directory (checkpoints, logs)
#   -l 64 \                          # latent dimensionality
#   -A path/to/aws_creds.json        # JSON file with { "access_key": "...", "secret_access_key": "..." }
#   --epochs 50 \                    # total number of epochs
#   --batch-size 16 \                # batch size
#   --lr 1e-3 \                      # learning rate
#   --weight-decay 1e-5 \            # optimizer weight decay
#   --scheduler-step 10 \            # LR scheduler step size
#   --scheduler-gamma 0.5 \          # LR scheduler decay factor
#   --slow-weight 0.1 \              # weight on time‐aware loss term
#   --early-stopping 5               # patience for early stopping

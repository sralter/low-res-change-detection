# inference_streamlit.py
# FOR STREAMLIT
# Inference script: load data, build inputs with metadata helpers, and run VAE change detection

from prepare_dataset import ZarrChangePairDataset, open_zarr
from vae import FCVAE as VAE
from inspect_dataset import list_vars, compute_band_summary, compute_time_summary, time_indices_to_dates

import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.backends.backend_pdf import PdfPages
import pymaap
pymaap.init_general_logger(console_level=logging.INFO)
from scipy.ndimage import zoom
import s3fs
from typing import Callable

def load_climatology(bucket: str, folder: str, geohash: str, 
                     aws_creds: dict | None = None) -> xr.DataArray:
    logging.info("loading climatology, please be patient...")
    path = f"s3://{bucket}/{folder}/{geohash}/{geohash}_climatology"
    if aws_creds:
        s3 = s3fs.S3FileSystem(
            anon=False,
            key=aws_creds["access_key"],
            secret=aws_creds['secret_access_key']
        )
    else:
        s3 = s3fs.S3FileSystem(anon=True)
    mapper = s3.get_mapper(path)
    return {
        'seasonal_ndvi': xr.open_dataarray(mapper / 'seasonal_ndvi.nc'),
        'monthly_mu':    xr.open_dataarray(mapper / 'monthly_ndvi_mean.nc'),
        'monthly_sigma': xr.open_dataarray(mapper / 'monthly_ndvi_std.nc'),
        'band_stats':    (json.load((mapper / 'band_stats.json').open())
                          if (mapper / 'band_stats.json').exists() else None)
    }  

def extract_patch(ds_arr, time_dim, lat_dim, lon_dim, idx, r, c, ps):
    """Return raw spectral patch of shape (B, H, W)."""
    logging.debug('extracting patch')
    return ds_arr.isel({time_dim: idx,
                        lat_dim: slice(r, r+ps),
                        lon_dim: slice(c, c+ps)}).values

def build_metadata(r, c, ps, times, idx0, idx1, clim, lat_dim, lon_dim):
    """
    Build metadata channels for t0 & t1: delta-time, DOY, seasonal NDVI, z-NDVI.
    Returns two arrays of shape (M, H, W).
    """
    logging.info('building metadata')
    H, W = ps, ps
    # delta time
    dt = (times[idx1] - times[idx0]) / pd.Timedelta(days=1)
    delta0 = np.zeros((1, H, W), dtype=np.float32)
    delta1 = np.full((1, H, W), dt, dtype=np.float32)
    # DOY
    doy_val = times[idx0].dayofyear
    doy0 = np.full((1, H, W), doy_val / 365.0, dtype=np.float32)
    # seasonal NDVI
    if clim['seasonal_ndvi'] is not None:
        s1 = clim['seasonal_ndvi'].sel(doy=doy_val, method='nearest')
        s1 = s1.isel(lat=slice(r, r+ps), lon=slice(c, c+ps)).values[None]
        s1 = np.nan_to_num(s1, nan=0.0)
    else:
        s1 = np.zeros((1, H, W), dtype=np.float32)
    s0 = np.zeros_like(s1)
    # z-NDVI
    if clim['monthly_mu'] is not None and clim['monthly_sigma'] is not None:
        m = clim['monthly_mu'].sel(month=times[idx0].month, method='nearest')
        sd = clim['monthly_sigma'].sel(month=times[idx0].month, method='nearest')
        m = m.isel(lat=slice(r, r+ps), lon=slice(c, c+ps)).values
        sd = sd.isel(lat=slice(r, r+ps), lon=slice(c, c+ps)).values
        m = np.nan_to_num(m, nan=0.0)
        sd = np.nan_to_num(sd, nan=1.0)  # avoid division by zero
        if 'ndvi' in clim['band_stats'] or True:
            # assume ndvi is first band if present
            z1 = ((m - m) / (sd + 1e-6))[None]  # placeholder
        else:
            z1 = np.zeros((1, H, W), dtype=np.float32)
    else:
        z1 = np.zeros((1, H, W), dtype=np.float32)
    z0 = np.zeros_like(z1)
    # stack metadata channels
    meta0 = np.concatenate([delta0, doy0, s0, z0], axis=0)
    meta1 = np.concatenate([delta1, doy0, s1, z1], axis=0)
    return meta0, meta1

def prepare_inputs(raw0, raw1, meta0, meta1, band_stats, device):
    """Concatenate raw and meta, normalize, and return tensors."""
    logging.debug('preparing inputs')
    inp0 = np.concatenate([raw0, meta0], axis=0)
    inp1 = np.concatenate([raw1, meta1], axis=0)
    x0 = torch.from_numpy(inp0).unsqueeze(0).float().to(device)
    x1 = torch.from_numpy(inp1).unsqueeze(0).float().to(device)
    if band_stats:
        # apply per-band z-score normalization using stats dict
        eps = 1e-6
        for idx, (band, stats) in enumerate(band_stats.items()):
            m = stats['mean']
            s = stats['std']
            if s == 0 or np.isnan(m) or np.isnan(s):
                logging.warning(f"Bad stats for band {band}: mean={m}, std={s}")
            x0[:, idx:idx+1, :, :] = (x0[:, idx:idx+1, :, :] - m) / (s + eps)
            x1[:, idx:idx+1, :, :] = (x1[:, idx:idx+1, :, :] - m) / (s + eps)
    return x0, x1

def to_rgb(img_tensor):
    """Simple min/max stretch on first 3 channels to [0,1] for display."""
    arr = img_tensor[:3].cpu().numpy()
    # H×W×3
    arr = np.clip((arr - arr.min())/(arr.max()-arr.min()+1e-6), 0, 1)
    return arr.transpose(1,2,0)

def stretch_img(img, p_low=2, p_high=98):
    """Percentile stretch for visualization."""
    logging.info('stretch image')
    lo, hi = np.percentile(img, (p_low, p_high))
    if hi - lo < 1e-3:  # low contrast image
        logging.info("Low-contrast image")
        return np.zeros_like(img)
    logging.info("Image has good contrast")
    return np.clip((img - lo) / (hi - lo), 0, 1)

def save_rgb(ds_arr, time_dim, lat_dim, lon_dim, idx, bands, run_dir, rgb_bands):
    logging.info('save rgb')
    raw = extract_patch(ds_arr, time_dim, lat_dim, lon_dim, idx, 0, 0, ds_arr.sizes[lat_dim])

    # debug
    if np.isnan(raw).all():
        logging.warning(f"Entire raw input at t{idx} is NaN")
    elif np.max(raw) == 0:
        logging.warning(f"All-zero raw input at t{idx}")
    else:
        logging.info(f"t{idx} raw patch stats — min: {np.min(raw):.2f}, max: {np.max(raw):.2f}")

    img = raw[rgb_bands].transpose(1,2,0)
    for c in range(3): 
        img[...,c] = stretch_img(img[...,c])

    logging.debug(f"t{idx} RGB band {rgb_bands}: pre-stretch min/max = {[raw[b].min() for b in rgb_bands]}, {[raw[b].max() for b in rgb_bands]}")

    plt.imsave(run_dir / f"t{idx}.png", img)

def plot_full_and_patch(
    ds: xr.Dataset,
    bands: list[str],
    t0: int,
    t1: int,
    r0: int, c0: int, ps: int,
    tdim: str, lat_dim: str, lon_dim: str,
    out_path: Path#,
    # geohash: str,
    # total: int
):
    # 1) grab lon/lat coords for full‐scene extent
    lons = ds[lon_dim].values
    lats = ds[lat_dim].values
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    # 2) helper to stretch & stack
    def make_rgb(arr3d):
        # arr3d: (3, H, W) → (H, W, 3)
        img = np.moveaxis(arr3d, 0, -1)
        lo, hi = np.percentile(img, (2,98))
        return np.clip((img - lo)/(hi - lo + 1e-6), 0, 1)

    # 3) full‐frame t0/t1
    full0 = np.stack([ds[b].isel({tdim:t0}).values for b in bands], axis=0)
    full1 = np.stack([ds[b].isel({tdim:t1}).values for b in bands], axis=0)
    rgb_full0 = make_rgb(full0)
    rgb_full1 = make_rgb(full1)

    # 4) patch at (r0,c0)
    patch0 = np.stack([
        ds[b].isel({tdim:t0,
                    lat_dim: slice(r0,r0+ps),
                    lon_dim: slice(c0,c0+ps)}).values
        for b in bands
    ], axis=0)
    patch1 = np.stack([
        ds[b].isel({tdim:t1,
                    lat_dim: slice(r0,r0+ps),
                    lon_dim: slice(c0,c0+ps)}).values
        for b in bands
    ], axis=0)
    rgb_patch0 = make_rgb(patch0)
    rgb_patch1 = make_rgb(patch1)

    # 5) compute patch corner in lon/lat for rectangle
    lon0, lon1 = float(lons[c0]), float(lons[min(c0+ps-1,len(lons)-1)])
    lat0, lat1 = float(lats[min(r0+ps-1,len(lats)-1)]), float(lats[r0])

    # 6) plot 2×2
    fig, axes = plt.subplots(2,2, figsize=(10,10))
    # fig.suptitle(
    #     f"True-Color\nGeohash: {geohash}\nIndex: {idx0}/{total}\nPatch Location: {patch_loc+1}/{tiles_per_slice}\nDate: {date0} → {date1}",
    #     y=1.01,
    #     fontsize=14
    # )
    for ax, img, title in zip(axes[0], (rgb_full0, rgb_full1), ("t₀ full", "t₁ full")):
        ax.imshow(img, extent=extent, origin="upper")
        ax.set_title(title)
        ax.add_patch(Rectangle(
            (lon0, lat0),
            lon1-lon0, lat1-lat0,
            edgecolor="red", facecolor="none", linewidth=2
        ))
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")

    for ax, img, title in zip(axes[1], (rgb_patch0, rgb_patch1), ("t₀ patch", "t₁ patch")):
        ax.imshow(img, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("Patch x")
        ax.set_ylabel("Patch y")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def make_report(out_dir, t0_img, t1_img,
                # heatmap, 
                # mask,
                heatmap_full, mask_full,
                date0_idx, date1_idx, ds,
                geohash: str):
    pdf_path = Path(out_dir) / "report.pdf"
    # pdf_path = Path(out_dir)/f"report_{date0_idx}_{date1_idx}.pdf"
    date0, date1 = time_indices_to_dates(ds=ds, idx0=date0_idx, idx1=date1_idx)
    with PdfPages(pdf_path) as pdf:
        # ── Page 1: raw snapshots ─────────────────────────────────────
        fig, axes = plt.subplots(1,2,figsize=(8,4))
        fig.suptitle(f"Geohash: {geohash}", y=1.00, fontsize=14)
        axes[0].imshow(t0_img)
        axes[0].set_title(f"t₀ (idx={date0_idx}, date={date0})")
        axes[0].axis("off")
        axes[1].imshow(t1_img)
        axes[1].set_title(f"t₁ (idx={date1_idx}, date={date1})")
        axes[1].axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: full‐scene heatmap alone ───────────────────────────
        fig, ax = plt.subplots(figsize=(6,6))
        fig.suptitle(f"Geohash: {geohash}", y=1.00, fontsize=14)
        im = ax.imshow(heatmap_full, cmap="hot")
        ax.set_title("Full-scene Δμ heatmap")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: full‐scene mask alone ──────────────────────────────
        fig, ax = plt.subplots(figsize=(6,6))
        fig.suptitle(f"Geohash: {geohash}", y=1.00, fontsize=14)
        ax.imshow(mask_full, cmap="gray")
        ax.set_title("Full-scene change mask")
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: overlays ───────────────────────────────────────────
        fig, axes = plt.subplots(1,2, figsize=(12,6))
        fig.suptitle(f"Geohash: {geohash}", y=1.00, fontsize=14)
        # heatmap overlay on t₀
        axes[0].imshow(t0_img)
        axes[0].imshow(heatmap_full, cmap="hot", alpha=0.5)
        axes[0].set_title(f"Δμ heatmap over t₀ ({date0})")
        axes[0].axis("off")
        # mask overlay on t₁
        axes[1].imshow(t1_img)
        axes[1].imshow(mask_full, cmap="Greens", alpha=0.6)
        axes[1].set_title(f"Change mask over t₁ {date1}")
        axes[1].axis("off")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 5: numeric summary ────────────────────────────────────
        mean_h, std_h, max_h = heatmap_full.mean(), heatmap_full.std(), heatmap_full.max()
        # you can grab the threshold you used, if you stored it somewhere;
        # or recompute: thresh = mean_h + 2*std_h
        thresh = mean_h + 2*std_h
        pct_changed    = mask_full.sum()/mask_full.size * 100
        stats = [
            ("Mean Δμ", f"{mean_h:.3f}"),
            ("Std Δμ",  f"{std_h:.3f}"),
            ("Max Δμ",  f"{max_h:.3f}"),
            ("Threshold", f"{thresh:.3f}"),
            ("% patches > thr", f"{pct_changed:.2f}%"),
            ("Patch grid size", f"{heatmap_full.shape[0]} x {heatmap_full.shape[1]}"),
        ]
        try:
            stats_df = pd.DataFrame(data=stats)
            stats_df_path = out_dir / "summary_statistics.csv"
            stats_df.to_csv(stats_df_path)
            logging.info(f"Saved statistics summary table to: {stats_df_path}")
        except Exception as e:
            logging.error(f"Error in saving statistics summary table: {e}")
        fig, ax = plt.subplots(figsize=(6,4))
        fig.suptitle(f"Geohash: {geohash}", y=1.00, fontsize=14)
        ax.axis("off")
        y = 0.9
        ax.text(0, y, "Change Detection Summary", fontsize=14, weight="bold")
        for label, val in stats:
            y -= 0.12
            ax.text(0, y, f"{label:20s}: {val}", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved PDF report to {pdf_path}")

    return str(pdf_path)

def to_time_indices(ds, date_strs):
    times = pd.to_datetime(ds.time.values)
    def str2idx(s: str):
        sl = s.lower()
        if sl == "first": return 0
        if sl == "last":  return len(times)-1
        return int((np.abs(times - np.datetime64(s))).argmin())
    i0, i1 = str2idx(date_strs[0]), str2idx(date_strs[1])
    if i1 < i0:
        i0, i1 = i1, i0
    # clamp
    i0 = max(0, min(i0, len(times)-2))
    i1 = max(i0+1, min(i1, len(times)-1))
    return i0, i1

def parse_args():
    p = argparse.ArgumentParser(
        description="Full-scene change-detection inference with a trained VAE"
    )
    p.add_argument("--bucket",   "-b", required=True, help="S3 bucket name")
    p.add_argument("--folder",   "-f", required=True, help="Folder within bucket")
    p.add_argument("--geohash",  "-g", required=True,
                   help="5-char geohash code")
    p.add_argument("--model",    "-m", required=True,
                   help="Path to best_model.pt")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("-i", "--index", type=int,
                     help="Global patch index → next slice (idx, idx+1)")
    grp.add_argument("-d", "--dates", nargs=2, metavar=("DATE0","DATE1"),
                     help="Two dates (YYYY-MM-DD or first/last)")
    p.add_argument("--aws-creds-file","-A", required=True,
                   help="JSON with {'access_key':..., 'secret_access_key':...}")
    p.add_argument("--out", "-o", required=True,
                   help="Directory for outputs & report")
    p.add_argument(
        "--stage-zarr", action="store_true", dest="stage_zarr",
        help="If set, download each Zarr locally before training."
    )
    p.add_argument(
        "--no-stage-zarr", action="store_false", dest="stage_zarr",
        help="Do *not* download Zarrs locally; stream from S3."
    )
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--stride",     type=int, default=None,
                   help="If set, override auto-computed stride (in pixels)")
    p.add_argument("--out-size", "-os", type=int, default=None,
                   help="Desired number of patches per dimension in the output heatmap")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--latent-dim", "-l", type=int, help=( # default=64
        "Must match the latent-dim used at train time"
    ))
    p.add_argument("--threshold",  type=float, default=None,
                   help="If unset, will use mean+2·std of heatmap")
    p.add_argument("--num-workers", type=int, default=0) # or 4 if locally run, not EC2
    return p.parse_args()

def run_inference_for_geohash(
    geohash: str,
    date0_str: str,
    date1_str: str,
    bucket: str = "rgc-zarr-store",
    folder: str = "data",
    model_path: str = "best_model.pt",
    aws_creds_path: str = "secrets/aws_rgc-zarr-store.json",
    out_dir: str = "inference_outputs",
    patch_size: int = 128,
    stride: int = None,
    out_size: int = None,
    batch_size: int = 16,
    latent_dim: int = None,
    threshold: float = None,
    num_workers: int = 0,
    stage_zarr: bool = False,
    progress_callback: Callable | None = None,
):
    """
    Streamlit‐friendly inference pipeline:
      • Streams the Zarr off S3 (using aws_creds_path),
      • writes all outputs locally under `out_dir/<geohash>_<date0>_<date1>/`,
      • returns a dict of local-filepaths so you can display or download them.
    
    Changes from your previous version:
      - `model_path` is now a local filesystem path (no S3FS).
      - We still use aws_creds_path + s3fs for opening the Zarr on S3.
      - All references to loading the model from S3 have been removed.
    """
    logging.info("Starting Streamlit-callable inference…")

    total_steps = 21
    step = 0

    # ─── 1) Load AWS credentials for Zarr‐streaming (we still pull the Zarr from S3)
    with open(aws_creds_path, "r") as f:
        aws_creds = json.load(f)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # Create an S3FileSystem to pass to open_zarr (so that xarray can stream from S3)
    s3 = s3fs.S3FileSystem(
        anon=False,
        key=aws_creds["access_key"],
        secret=aws_creds["secret_access_key"],
        client_kwargs={"region_name": "us-east-1"},  # adjust region if needed
    )
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 2) Build the S3‐backed zarr path and open it to figure out time indices
    zarr_s3_path = f"s3://{bucket}/{folder}/{geohash}/{geohash}_zarr_streamlit"

    ds_for_dates = open_zarr(
        zarr_s3_path, consolidated=True, aws_creds=aws_creds
    )
    logging.info(
        "Static‐band summary:\n%s",
        compute_band_summary(ds_for_dates).to_string(),
    )

    time_dim_name = next(dim for dim in ds_for_dates.dims if "time" in dim.lower())
    logging.info(
        "Time‐axis summary:\n%s",
        compute_time_summary(ds_for_dates[time_dim_name]).to_string(),
    )
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 3) Convert the input date strings into integer indices
    idx0, idx1 = to_time_indices(ds_for_dates, [date0_str, date1_str])
    logging.info(f"Selected time indices: {idx0} → {idx1}")
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 4) Re‐open the full Zarr just to stack all bands into one xarray
    full_ds = open_zarr(
        zarr_s3_path, consolidated=True, aws_creds=aws_creds
    )
    bands_list = ["R", "G", "B", "NDVI", "elevation", "slope", "aspect", "aspect_mask"]

    ds_arr = (
        xr.concat([full_ds[v] for v in bands_list], dim="band", coords="minimal", compat="override")
          .assign_coords(band=bands_list)
          .compute()
    )

    # Identify which dims are “time”, “lat”, “lon”
    time_dim = next(d for d in full_ds.dims if "time" in d.lower())
    lat_dim  = next(d for d in full_ds.dims if "lat" in d.lower() or "y" in d.lower())
    lon_dim  = next(d for d in full_ds.dims if "lon" in d.lower() or "x" in d.lower())
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 5) Create a local folder for output files:
    out_base = Path(out_dir) / f"{geohash}_{date0_str.replace('-','')}_{date1_str.replace('-','')}"
    out_base.mkdir(parents=True, exist_ok=True)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 6) Plot + save the full & patch RGB images (so the user sees “before/after”)
    plot_full_and_patch(
        ds=full_ds,
        bands=["R", "G", "B"],
        t0=idx0,
        t1=idx1,
        r0=0,
        c0=0,
        ps=patch_size,
        tdim=time_dim,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        out_path=out_base / "rgb_before_after.png",
    )
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 7) If out_size was provided, auto‐compute stride
    if out_size is not None:
        H, W = ds_for_dates.sizes[lat_dim], ds_for_dates.sizes[lon_dim]
        stride_y = max(1, (H - patch_size) // (out_size - 1))
        stride_x = max(1, (W - patch_size) // (out_size - 1))
        stride = min(stride_y, stride_x)
        logging.info(f"Auto‐computed stride={stride} for out_size={out_size}")
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 8) Also save pure‐RGB “T0” and “T1” images for the Streamlit UI
    rgb_bands = [bands_list.index(c) for c in ("R", "G", "B")]
    save_rgb(ds_arr, time_dim, lat_dim, lon_dim, idx0, bands_list, out_base, rgb_bands)
    save_rgb(ds_arr, time_dim, lat_dim, lon_dim, idx1, bands_list, out_base, rgb_bands)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 9) Build the one‐pair PyTorch DataLoader
    ds = ZarrChangePairDataset(
        zarr_path=zarr_s3_path,
        patch_size=patch_size,
        stride=stride or patch_size,
        bands=bands_list,
        seasonal_ndvi_path=f"s3://{bucket}/{folder}/{geohash}/{geohash}_climatology/seasonal_ndvi.nc",
        monthly_mu_path   =f"s3://{bucket}/{folder}/{geohash}/{geohash}_climatology/monthly_ndvi_mean.nc",
        monthly_sigma_path=f"s3://{bucket}/{folder}/{geohash}/{geohash}_climatology/monthly_ndvi_std.nc",
        aws_creds=aws_creds,
        stage=stage_zarr,
        time_indices=[(idx0, idx1)],
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 10) Load the VAE checkpoint from a local path (no S3FS here)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    ckpt = torch.load(model_path, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)

    if latent_dim is None:
        latent_key = next((k for k in sd.keys() if k.endswith("encoder.conv_mu.weight")), None)
        if latent_key is None:
            raise KeyError(
                "Could not find 'encoder.conv_mu.weight' in checkpoint keys:\n" + "\n".join(sd.keys())
            )
        latent_dim = sd[latent_key].shape[0]

    model = VAE(in_channels=13, latent_dim=latent_dim, use_time_embed=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 11) Allocate a “heatmap canvas” + counter, then loop over patches:
    H_full = ds_for_dates.sizes[lat_dim]
    W_full = ds_for_dates.sizes[lon_dim]
    patch  = patch_size
    stride = stride or patch

    rows = list(range(0, H_full - patch + 1, stride))
    if rows[-1] + patch < H_full:
        rows.append(H_full - patch)
    cols = list(range(0, W_full - patch + 1, stride))
    if cols[-1] + patch < W_full:
        cols.append(W_full - patch)

    heatmap_full = np.zeros((H_full, W_full), dtype=float)
    count_map    = np.zeros((H_full, W_full), dtype=int)

    times = pd.to_datetime(ds_for_dates[time_dim].values)
    total_tiles = len(rows) * len(cols)
    tile_idx = 0
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    with torch.no_grad():
        for r in rows:
            for c in cols:
                # tile_idx += 1
                # # update Streamlit progress bar via callback
                # if progress_callback is not None:
                #     progress_callback(tile_idx / float(total_tiles))

                raw0 = extract_patch(ds_arr, time_dim, lat_dim, lon_dim, idx0, r, c, patch)
                raw1 = extract_patch(ds_arr, time_dim, lat_dim, lon_dim, idx1, r, c, patch)

                if np.isnan(raw0).all() or np.isnan(raw1).all():
                    continue

                raw0 = np.nan_to_num(raw0, nan=0.0)
                raw1 = np.nan_to_num(raw1, nan=0.0)

                # convert “aspect” band → sin/cos
                aspect0 = raw0[6]
                sin0    = np.sin(np.deg2rad(aspect0))
                cos0    = np.cos(np.deg2rad(aspect0))
                raw0 = np.concatenate([raw0[0:6], sin0[None], cos0[None], raw0[7:8]], axis=0)

                aspect1 = raw1[6]
                sin1    = np.sin(np.deg2rad(aspect1))
                cos1    = np.cos(np.deg2rad(aspect1))
                raw1 = np.concatenate([raw1[0:6], sin1[None], cos1[None], raw1[7:8]], axis=0)

                clim_dummy = {
                    "seasonal_ndvi": None,
                    "monthly_mu": None,
                    "monthly_sigma": None,
                    "band_stats": None,
                }
                meta0, meta1 = build_metadata(r, c, patch, times, idx0, idx1, clim_dummy, lat_dim, lon_dim)
                meta0 = np.nan_to_num(meta0, nan=0.0)
                meta1 = np.nan_to_num(meta1, nan=0.0)

                x0, x1 = prepare_inputs(raw0, raw1, meta0, meta1, band_stats=None, device=device)

                mu0, _, _ = model.encoder(x0)
                dt_scalar = float((times[idx1] - times[idx0]) / pd.Timedelta(days=1))
                dt_tensor = torch.tensor([[dt_scalar]], dtype=torch.float32, device=device)
                _, mu1, _ = model(x1, delta_t=dt_tensor)

                diff_lat = torch.norm(mu1 - mu0, dim=1, keepdim=True)
                diff_128 = torch.nn.functional.interpolate(
                    diff_lat, size=(patch, patch), mode="bilinear", align_corners=False
                )
                diff_128 = diff_128.squeeze(0).squeeze(0).cpu().numpy()

                heatmap_full[r : r + patch, c : c + patch] += diff_128
                count_map[  r : r + patch, c : c + patch] += 1
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 12) Average overlapping contributions
    nonzero_mask = count_map > 0
    heatmap_full[nonzero_mask] /= count_map[nonzero_mask]
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 13) Threshold → binary mask
    if threshold is None:
        thresh_val = heatmap_full.mean() + 2.0 * heatmap_full.std()
    else:
        thresh_val = threshold
    mask_full = (heatmap_full > thresh_val).astype(np.uint8)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 14) Save heatmap + mask locally
    heatmap_png = out_base / "heatmap_full.png"
    mask_png    = out_base / "mask_full.png"
    np.save(out_base / "heatmap_full.npy", heatmap_full)
    plt.imsave(heatmap_png, heatmap_full, cmap="hot")
    np.save(out_base / "mask_full.npy", mask_full)
    plt.imsave(mask_png, mask_full, cmap="gray")
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 15) Load the saved RGB “t0”/“t1” (they were already written in step 8)
    t0_img = plt.imread(out_base / f"t{idx0}.png")
    t1_img = plt.imread(out_base / f"t{idx1}.png")
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # (9) Build TWO additional overlay images and save them:
    #     – overlay_heatmap_t0.png (t₀ + Δμ‐heatmap on top)
    #     – overlay_mask_t1.png   (t₁ + binary‐mask on top)

    # Overlay 1: Δμ heatmap over t₀
    overlay0_path = out_base / "overlay_heatmap_t0.png"
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(t0_img)
    ax.imshow(heatmap_full, cmap="hot", alpha=0.5)
    ax.axis("off")
    plt.savefig(overlay0_path, bbox_inches="tight")
    plt.close(fig)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # Overlay 2: change mask over t₁
    overlay1_path = out_base / "overlay_mask_t1.png"
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(t1_img)
    ax.imshow(mask_full, cmap="Greens", alpha=0.6)
    ax.axis("off")
    plt.savefig(overlay1_path, bbox_inches="tight")
    plt.close(fig)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 16) Build a PDF report locally
    report_pdf = make_report(
        out_base,
        t0_img,
        t1_img,
        heatmap_full=heatmap_full,
        mask_full=mask_full,
        date0_idx=idx0,
        date1_idx=idx1,
        ds=ds_for_dates,
        geohash=geohash,
    )
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 17) The “summary_statistics.csv” is written by make_report() itself
    stats_csv = out_base / "summary_statistics.csv"
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    logging.info("Inference complete. Outputs in %s", out_base)

    return {
        "rgb_t0":               str((out_base / f"t{idx0}.png").resolve()),
        "rgb_t1":               str((out_base / f"t{idx1}.png").resolve()),
        "heatmap":              str(heatmap_png.resolve()),
        "mask":                 str(mask_png.resolve()),
        "overlay_heatmap_t0":   str(overlay0_path.resolve()),
        "overlay_mask_t1":      str(overlay1_path.resolve()),
        "report_pdf":           str(Path(report_pdf).resolve()),
        "stats_csv":            str(stats_csv.resolve()),
    }

def main():
    logging.info("Starting Inference script...")
    args = parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() elif '' else 'cpu')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.aws_creds_file) as f:
        aws_creds = json.load(f)

    # a quick inspect
    logging.info("=== Dataset inspection ===")
    list_vars(args.aws_creds_file, args.bucket, args.folder, args.geohash)
    zarr_path = f"s3://{args.bucket}/{args.folder}/{args.geohash}/{args.geohash}_zarr"
    logging.info(f"Opening Zarr from {zarr_path}...")
    ds_for_dates = open_zarr(zarr_path,
                    consolidated=True, aws_creds=aws_creds)
    logging.info("Static-band summary:\n%s", compute_band_summary(ds_for_dates).to_string())
    tdim = next(d for d in ds_for_dates.dims if "time" in d.lower())
    logging.info("Time-axis summary:\n%s", compute_time_summary(ds_for_dates[tdim]).to_string())

    # figure out which slice‐pair to run
    if args.index is not None:
        idx0, idx1 = args.index, args.index + 1
        max_t = ds_for_dates.sizes[tdim] - 1
        idx0 = min(max(idx0, 0), max_t-1)
        idx1 = idx0 + 1
        idx0_date, idx1_date = to_time_indices(ds_for_dates, args.dates)
    else:
        idx0, idx1 = to_time_indices(ds_for_dates, args.dates)
        idx0_date, idx1_date = to_time_indices(ds_for_dates, args.dates)
    # ds_for_dates = None

    times = pd.to_datetime(ds_for_dates[tdim].values)
    dt_scalar = float((times[idx1] - times[idx0]) / pd.Timedelta(days=1))

    logging.info(f"Running inference for time indices: {idx0} ({idx0_date}) → {idx1} ({idx1_date})")

    logging.info("Load full scene for both times in pair")
    full_path = f"s3://{args.bucket}/{args.folder}/{args.geohash}/{args.geohash}_zarr"
    full_ds = open_zarr(full_path, consolidated=True, aws_creds=aws_creds)
    # pick your bands in order:
    bands_list = ["R","G","B","NDVI","elevation","slope","aspect","aspect_mask"]
    ds_arr = xr.concat(
        [full_ds[v] for v in bands_list],
        dim="band", coords="minimal", compat="override"
    ).assign_coords(band=bands_list).compute()
    time_dim = next(d for d in full_ds.dims if "time" in d.lower())
    lat_dim  = next(d for d in full_ds.dims if "lat"  in d.lower() or "y" in d.lower())
    lon_dim  = next(d for d in full_ds.dims if "lon"  in d.lower() or "x" in d.lower())

    # get full image and patch images of t0 and t1
    # total = (len(ds[tdim])) * R * C
    plot_full_and_patch(
        ds=full_ds,
        bands=["R", "G", "B"],
        t0=idx0, t1=idx1,
        r0=0, c0=0, ps=args.patch_size,
        tdim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim,
        out_path=out_dir/"rgb_before_after.png"#,
        # geohash=args.geohash,
        # total=total
    )

    # —— auto-compute stride if user gave --out-size
    if args.out_size:
        H, W = ds_for_dates.sizes[lat_dim], ds_for_dates.sizes[lon_dim]
        # how many steps between patches to get ~out_size points:
        # (H - patch)/(N-1) ≃ stride_y
        stride_y = max(1, (H - args.patch_size) // (args.out_size - 1))
        stride_x = max(1, (W - args.patch_size) // (args.out_size - 1))
        # use the smaller so you get at least that many patches in both dims
        args.stride = min(stride_y, stride_x)
        logging.info(f"Auto-computed stride={args.stride} for target out-size={args.out_size}")

    # now write out two big RGB PNGs:
    rgb_bands = [bands_list.index(c) for c in ("R", "G", "B")]
    save_rgb(ds_arr, time_dim, lat_dim, lon_dim, idx0, bands_list, out_dir, rgb_bands)
    save_rgb(ds_arr, time_dim, lat_dim, lon_dim, idx1, bands_list, out_dir, rgb_bands)

    logging.info("Building one-pair dataset...")
    ds = ZarrChangePairDataset(
        zarr_path=f"s3://{args.bucket}/{args.folder}/{args.geohash}/{args.geohash}_zarr",
        patch_size=args.patch_size,
        stride    = args.stride or args.patch_size,
        bands     = ["R","G","B","NDVI","elevation","slope","aspect","aspect_mask"],
        seasonal_ndvi_path=f"s3://{args.bucket}/{args.folder}/{args.geohash}/{args.geohash}_climatology/seasonal_ndvi.nc",
        monthly_mu_path   =f"s3://{args.bucket}/{args.folder}/{args.geohash}/{args.geohash}_climatology/monthly_ndvi_mean.nc",
        monthly_sigma_path=f"s3://{args.bucket}/{args.folder}/{args.geohash}/{args.geohash}_climatology/monthly_ndvi_std.nc",
        aws_creds=aws_creds,
        stage=args.stage_zarr,
        time_indices=[(idx0, idx1)]
    )

    logging.info("Instantiating loader...")
    loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)

    logging.info(f"Loading model {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    # state-dict might be under "model_state_dict" or at the top level:
    sd = ckpt.get("model_state_dict", ckpt)

    latent_key = None
    for k in sd.keys():
        if k.endswith("encoder.conv_mu.weight"):
            latent_key = k
            break

    if not args.latent_dim:
        if latent_key is None:
            raise KeyError(
                f"could not find 'encoder.conv_mu.weight' in checkpoint. Available keys:\n"
                + "\n".join(sd.keys())
            )
        # The conv_mu.weight tensor shape is [latent_dim, C_in, k, k].
        # We only need the first dimension (latent_dim).
        latent_dim = sd[latent_key].shape[0]
    else:
        latent_dim = args.latent_dim
    # number of raw channels equals:
    # ["R","G","B","NDVI","elevation","slope","aspect","aspect_mask"]
    # but then we added more and made aspect into two to handle 350degrees ~ ~ 10 degrees:
    # ["R","G","B","NDVI","elevation","slope","aspect_sin", "aspect_cos", "aspect_mask", ∆t, DOY, seasonal_NDVI, z_NDVI]
    # 13 total for in_channels
    model = VAE(in_channels=13, latent_dim=latent_dim, use_time_embed=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # ── Now build a FULL‐SCENE 500×500 heatmap by upsampling each 128×128 patch ──
    ds_arr = xr.concat(
        [full_ds[v] for v in bands_list],
        dim="band", coords="minimal", compat="override"
    ).assign_coords(band=bands_list).compute()
    
    H_full = ds_for_dates.sizes[lat_dim]
    W_full = ds_for_dates.sizes[lon_dim]
    patch  = args.patch_size
    stride = args.stride or patch

    # build a list of top‐left corners so we tile the entire H×W without
    # missing any pixels (and without “upsampling” at the end):
    rows = list(range(0, H_full - patch + 1, stride))
    if rows[-1] + patch < H_full:
        rows.append(H_full - patch)
    cols = list(range(0, W_full - patch + 1, stride))
    if cols[-1] + patch < W_full:
        cols.append(W_full - patch)

    # allocate a “canvas” for the full‐scene heatmap, and a count‐map to average overlaps:
    heatmap_full = np.zeros((H_full, W_full), dtype=float)
    count_map    = np.zeros((H_full, W_full), dtype=int)

    with torch.no_grad():
        for r in rows:
            for c in cols:
                #
                # 1) extract one 8‐channel raw patch at t0 and one at t1
                #
                raw0 = extract_patch(
                    ds_arr, time_dim, lat_dim, lon_dim,
                    idx0, r, c, patch
                )  # shape == (8, patch, patch)
                raw1 = extract_patch(
                    ds_arr, time_dim, lat_dim, lon_dim,
                    idx1, r, c, patch
                )  # shape == (8, patch, patch)

                # 2) skip if either patch is entirely NaN
                if np.isnan(raw0).all() or np.isnan(raw1).all():
                    continue

                # 3) replace any remaining NaNs with zeros (so sin/cos won’t blow up)
                raw0 = np.nan_to_num(raw0, nan=0.0)
                raw1 = np.nan_to_num(raw1, nan=0.0)

                # 4) split off the single “aspect” band (index=6), compute sin/cos,
                #    and re‐stack so that “aspect” → “aspect_sin”, “aspect_cos”
                #  raw0 has channels indexed:
                #    0→R, 1→G, 2→B, 3→NDVI, 4→elevation, 5→slope,
                #    6→aspect (in degrees), 7→aspect_mask
                #
                aspect0 = raw0[6]  # shape (patch, patch)
                sin0    = np.sin(np.deg2rad(aspect0))
                cos0    = np.cos(np.deg2rad(aspect0))
                raw0 = np.concatenate([
                    raw0[0:6],   # R, G, B, NDVI, elevation, slope  → 6 ch
                    sin0[None],  # aspect_sin                         → 1 ch
                    cos0[None],  # aspect_cos                         → 1 ch
                    raw0[7:8],   # aspect_mask                        → 1 ch
                ], axis=0)
                # now raw0.shape == (9, patch, patch)

                aspect1 = raw1[6]
                sin1    = np.sin(np.deg2rad(aspect1))
                cos1    = np.cos(np.deg2rad(aspect1))
                raw1 = np.concatenate([
                    raw1[0:6],
                    sin1[None],
                    cos1[None],
                    raw1[7:8],
                ], axis=0)
                # now raw1.shape == (9, patch, patch)

                # 5) build the four metadata channels: Δt, DOY, seasonal NDVI, z‐NDVI
                #    (we pass in a dummy “clim” dict with None, so it will produce
                #    zero‐arrays for seasonal & z-NDVI).
                #
                #    You must have computed “times” earlier in main():
                #      times = pd.to_datetime(ds_for_dates.time.values)
                #
                #    and a small helper that turns (r,c,patch,...) into 4×patch×patch:
                #
                meta0, meta1 = build_metadata(
                    r, c, patch,
                    times, idx0, idx1,
                    clim={
                        "seasonal_ndvi": None,
                        "monthly_mu": None,
                        "monthly_sigma": None,
                        "band_stats": None
                    },
                    lat_dim=lat_dim,
                    lon_dim=lon_dim
                )
                # meta0/1 each have shape (4, patch, patch)
                meta0 = np.nan_to_num(meta0, nan=0.0)
                meta1 = np.nan_to_num(meta1, nan=0.0)

                #
                # 6) concatenate raw + meta → (13, patch, patch), then make torch tensors
                #
                x0, x1 = prepare_inputs(
                    raw0, raw1, meta0, meta1,
                    band_stats=None,
                    device=device
                )
                # now x0.shape == (1, 13, patch, patch)
                # and    x1.shape == (1, 13, patch, patch)

                # 7) run encoder(x0) and full VAE(x1, Δt) to get two latent maps:
                #    mu0.shape == (1, L, h_lat, w_lat),  for example (1, 32, 16, 16)
                #    mu1.shape == (1, L, h_lat, w_lat)
                #
                mu0, logvar0, _ = model.encoder(x0)

                # build Δt as a 1×1 tensor (in days):
                dt_scalar = (
                    ds_for_dates.time.values[idx1]
                    - ds_for_dates.time.values[idx0]
                ) / np.timedelta64(1, "D")
                dt = torch.tensor([[dt_scalar]],
                                  dtype=torch.float32,
                                  device=device)

                _, mu1, logvar1 = model(x1, delta_t=dt)

                # 8) compute a single “difference” map per latent:
                #    - first compute the channel-wise L2 norm over the (h_lat×w_lat) grid:
                diff_lat = torch.norm(mu1 - mu0, dim=1, keepdim=True)
                # diff_lat.shape == (1,1,h_lat,w_lat), e.g. (1,1,16,16)

                #    - then bilinear-upsample from (h_lat×w_lat) → (patch×patch):
                diff_128 = torch.nn.functional.interpolate(
                    diff_lat,
                    size=(patch, patch),
                    mode="bilinear",
                    align_corners=False
                )  # → (1,1, patch, patch)
                diff_128 = diff_128.squeeze(0).squeeze(0).cpu().numpy()
                # diff_128.shape == (patch, patch)

                # 9) “Tile” that 128×128 patch into our full 500×500 canvas.  Whenever
                #    multiple patches overlap, we accumulate and increment count_map:
                heatmap_full[r : r + patch, c : c + patch] += diff_128
                count_map[  r : r + patch, c : c + patch] += 1

    # 10) wherever count_map >0, divide by the count to average overlapping contributions:
    mask_nonzero = (count_map > 0)
    heatmap_full[mask_nonzero] /= count_map[mask_nonzero]

    # 11) apply a threshold (mean + 1·std by default):
    if args.threshold is None:
        thresh = heatmap_full.mean() + 1.0 * heatmap_full.std()
    else:
        thresh = args.threshold
    mask_full = (heatmap_full > thresh).astype(np.uint8)

    # 12) now “heatmap_full” is exactly H_full×W_full (≈500×500), with no extra
    #     upsampling at the end.  Save it:
    np.save(out_dir / "heatmap_full.npy", heatmap_full)
    plt.imsave(out_dir / "heatmap_full.png", heatmap_full, cmap="hot")
    np.save(out_dir / "mask_full.npy", mask_full)
    plt.imsave(out_dir / "mask_full.png", mask_full, cmap="gray")

    # ── PDF report (using full resolution maps now) ───────────────
    t0_img = plt.imread(out_dir/f"t{idx0}.png")
    t1_img = plt.imread(out_dir/f"t{idx1}.png")
    _ = make_report(
        out_dir,
        t0_img,
        t1_img,
        # heatmap=None,         # old patch‐grid not used
        # mask=None,            # old patch‐grid not used
        heatmap_full=heatmap_full,
        mask_full=mask_full,
        date0_idx=idx0_date,
        date1_idx=idx1_date,
        ds=ds_for_dates,
        geohash=args.geohash
    )

    logging.info("Done. Outputs located at: %s", out_dir)

if __name__ == '__main__':
    main()

# usage example:
# python inference.py \
#   --model results/run3/best_model.pt \
#   --zarr data/9vgmd.zarr \
#   --date0 2021-01-01 \
#   --date1 2021-02-01 \
#   --out outputs/9vgmd_change \
#   --patch-size 128 \
#   --stride 64 \
#   --rgb-bands 3 2 1 \
#   --threshold 1.5

"""
prepare_dataset.py

Utility for preparing change‐pair training datasets from Zarr stores.

Provides:
  - open_zarr: load a Zarr-backed xarray Dataset from local or S3
  - open_dataarray: load a small DataArray (e.g. seasonal NDVI) from local or S3
  - compute_global_band_ranges: aggregate per-geohash min/max into global ranges
  - ZarrChangePairDataset: a torch.utils.data.Dataset that lazily reads two-time-step
    patches (with climatology and metadata layers), normalizes bands, and returns
    (pair_tensor, Δt_tensor) for model training.

Supports optional S3 staging, per-geohash or global normalization, and efficient
worker-side opening for DataLoader multiprocessing.
"""

import json
import torch
from torch.utils.data import Dataset, get_worker_info
import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
import s3fs
import fsspec
from tempfile import TemporaryDirectory
import logging
from pymaap import init_general_logger
init_general_logger()

fs = s3fs.S3FileSystem()

def open_zarr(path: str,
              consolidated: bool,
              aws_creds: dict | None = None) -> xr.Dataset:
    """
    Open an xarray Dataset from a Zarr store on local disk or S3.

    If `path` begins with "s3://", initializes an S3FileSystem (with
    optional AWS credentials), obtains a mapper, and calls `xr.open_zarr`.
    Otherwise, calls `xr.open_zarr` directly on the local path.

    Args:
        path (str): Local file path or "s3://bucket/key" URI to the Zarr store.
        consolidated (bool): Whether the Zarr store uses consolidated metadata.
        aws_creds (dict | None): Optional AWS credentials with keys
            "access_key" and "secret_access_key".

    Returns:
        xr.Dataset: The opened xarray Dataset.
    """
    if path.startswith("s3://"):
        # build a fresh S3FileSystem with or without creds
        if aws_creds:
            s3 = s3fs.S3FileSystem(
                anon=False,
                key=aws_creds["access_key"],
                secret=aws_creds["secret_access_key"],
            )
        else:
            s3 = s3fs.S3FileSystem(anon=True)
        mapper = s3.get_mapper(path)   # no anon/creds here
        return xr.open_zarr(mapper, consolidated=consolidated)
    else:
        return xr.open_zarr(path)

def open_dataarray(path: str, aws_creds: dict | None = None) -> xr.DataArray:
    """
    Open an xarray DataArray (e.g., a NetCDF file) from disk or S3.

    For S3 URIs, stages the file into a temporary directory (retained until
    process exit) and then loads it with `xr.open_dataarray`. For local paths,
    calls `xr.open_dataarray` directly.

    Args:
        path (str): Local file path or "s3://bucket/key" URI to the .nc file.
        aws_creds (dict | None): Optional AWS credentials.

    Returns:
        xr.DataArray: The opened DataArray.
    """
    if path.startswith("s3://"):
        # download the .nc to a local temp file, then open it
        so: dict[str, str] = {}
        if aws_creds:
            so = {
                "anon": False,
                "key":   aws_creds["access_key"],
                "secret":aws_creds["secret_access_key"],
            }
        else:
            so = {'anon': True}
        fs = fsspec.filesystem("s3", **so)
        # strip the "s3://" prefix
        remote_key = path.removeprefix("s3://")

        # stage into its own TemporaryDirectory so it survives until program exit
        tmp = TemporaryDirectory(prefix="nc_stage_")
        local_path = Path(tmp.name) / Path(remote_key).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote_key, str(local_path))

        # keep the tempdir alive so file doesn't vanish
        open_dataarray._tempdirs.append(tmp)  # type: ignore[attr-defined]

        return xr.open_dataarray(str(local_path))
    else:
        return xr.open_dataarray(path)
# keep all of our tempdirs alive until the Python process exits
open_dataarray._tempdirs: list[TemporaryDirectory] = []  # type: ignore[attr-defined]

def compute_global_band_ranges(
    bucket: str,
    folder: str,
    geohashes: list[str],
    aws_creds: dict | None = None
) -> tuple[dict[str,float], dict[str,float]]:
    """
    Compute global min/max values for each band across multiple geohashes.

    Reads each `band_stats.json` at
    `s3://<bucket>/<folder>/<gh>/<gh>_climatology/band_stats.json`, then for
    each band finds the minimum of all per-geohash "min" entries and the maximum
    of all per-geohash "max" entries.

    Args:
        bucket (str): S3 bucket name (no "s3://" prefix).
        folder (str): S3 prefix under the bucket.
        geohashes (list[str]): List of 5-character geohash strings.
        aws_creds (dict | None): Optional AWS credentials.

    Returns:
        tuple[dict[str, float], dict[str, float]]:
            - global_min: Mapping band → smallest observed min.
            - global_max: Mapping band → largest observed max.
    """
    if aws_creds:
        fs_local = s3fs.S3FileSystem(
            anon=False,
            key=aws_creds["access_key"],
            secret=aws_creds["secret_access_key"]
        )
    else:
        fs_local = s3fs.S3FileSystem(anon=True)

    global_min: dict[str, float] = {}
    global_max: dict[str, float] = {}

    for gh in geohashes:
        stats_s3_key = f"{bucket}/{folder}/{gh}/{gh}_climatology/band_stats.json"
        try:
            with fs_local.open(stats_s3_key) as f_in:
                stats = json.load(f_in)
        except Exception as e:
            raise RuntimeError(f"Could not open band_stats for {gh} at {stats_s3_key}: {e}")

        for band, band_entry in stats.items():
            mn = float(band_entry["min"])
            mx = float(band_entry["max"])
            if band not in global_min:
                global_min[band] = mn
                global_max[band] = mx
            else:
                if mn < global_min[band]:
                    global_min[band] = mn
                if mx > global_max[band]:
                    global_max[band] = mx

    return global_min, global_max

class ZarrChangePairDataset(Dataset):
    """
    A PyTorch Dataset that lazily loads 2-frame change-pair patches from a Zarr store.

    Each __getitem__(idx) returns a tensor of shape (2, C_total, H, W) containing
    all normalized bands and metadata layers for two time slices (t0, t1) and a
    Δt scalar tensor. Supports optional staging of the Zarr locally, per-geohash
    or global min/max normalization, and on-the-fly opening in worker processes.
    """
    def __init__(
        self,
        zarr_path: str,
        patch_size: int,             # = 128
        stride: int                  = None, # if None, defaults to patch_size
        bands: list[str]             = None,
        time_indices: list[tuple[int,int]] = None,
        seasonal_ndvi_path: str      = None,
        monthly_mu_path: str         = None,
        monthly_sigma_path: str      = None,
        aws_creds: dict | None       = None,
        stage: bool                  = False,
        global_min_vals: np.ndarray  = None,  # min/max have shape = (len(bands),)
        global_max_vals: np.ndarray  = None):
        """
        Initialize the dataset by extracting metadata and optionally staging data.

        1. Optionally download the entire Zarr store locally.
        2. Immediately load climatology DataArrays (seasonal NDVI, monthly μ/σ).
        3. Open the Zarr once to read dims, time array, and compute sliding-window
           grid (rows, cols, total_tiles).
        4. Load per-band min/max (either provided globally or per-geohash JSON).
        5. Store all state needed for lazy in-worker opening.

        Args:
            zarr_path (str): Local path or "s3://…" URI to the Zarr store.
            patch_size (int): Height/width of the square patch.
            stride (int | None): Stride between patches; defaults to patch_size.
            bands (list[str] | None): Candidate band names; defaults to all 3D data_vars.
            time_indices (list[tuple[int,int]] | None): List of (t0, t1) pairs.
            seasonal_ndvi_path (str | None): URI to seasonal NDVI NetCDF.
            monthly_mu_path (str | None): URI to monthly mean NetCDF.
            monthly_sigma_path (str | None): URI to monthly std NetCDF.
            aws_creds (dict | None): AWS credentials for S3 access.
            stage (bool): If True and on S3, download entire Zarr locally.
            global_min_vals (np.ndarray | None): Global per-band mins (shape=(C,)).
            global_max_vals (np.ndarray | None): Global per-band maxs (shape=(C,)).

        We defer any Xarray/S3/Zarr opening until inside __getitem__.
        In __init__, we:
          1) (Optionally) stage the entire Zarr locally, if requested.
          2) Load the small climatology arrays (seasonal_ndvi, monthly_mu, monthly_sigma) immediately,
             since they are usually small and safe to pickle.
          3) Do a one-time open of the Zarr store into a temp `xr.Dataset` purely to extract:
             - time array (to compute `self.time` and `self.max_dt`)
             - the band names & dims
             - the H, W (lat/lon) sizes to compute patch grid (`self.rows, self.cols, self.row_starts, self.col_starts`)
          Then we close that temporary `xr.Dataset` and drop it. We retain only the metadata:
             - `self.time`, `self.max_dt`
             - `self.bands`, `self.time_dim`, `self.lat_dim`, `self.lon_dim`
             - `self.rows`, `self.cols`, `self.row_starts`, `self.col_starts`
             - `self.band_min/band_max` dictionaries
             - `self.total_tiles`
          Finally, we store `self._zarr_path` (string) and `self._aws_creds` so workers can re-open later.

        zarr_path:        local filesystem path (if staged) or "s3://..." URI for the zarr store
        patch_size:       height/width of each square patch
        stride:           stride between patches (defaults to patch_size)
        bands:            e.g. ["R","G","B","NDVI","elevation","slope","aspect","aspect_mask"]
        time_indices:     list of (t0, t1) index-pairs; if None, default to adjacent pairs
        seasonal_ndvi_path, monthly_mu_path, monthly_sigma_path: used for NDVI z‐scoring, unchanged
        aws_creds:        credentials dict if you need to access S3
        stage:            if True and zarr_path is S3, stage entire Zarr locally
        global_min_vals/global_max_vals: if provided, use these arrays (shape=(C,)) for min/max;
                                          otherwise fall back to reading per‐geohash band_stats.json.
        """
        # ─────────────────────────────────────────────────────────────────
        # 1) OPTIONALLY STAGE THE ENTIRE ZARR LOCALLY
        # ─────────────────────────────────────────────────────────────────
        if stage and zarr_path.startswith("s3://"):
			# create a tempdir that will auto-cleanup on process exit
            self._tmpdir = TemporaryDirectory(
                prefix=f"zarr_stage_{Path(zarr_path).stem}_"
            )
            local_root  = Path(self._tmpdir.name)
			# e.g. "my-bucket/path/to/geohash_zarr"
            bucket_key  = zarr_path.removeprefix("s3://")
            local_store = local_root / Path(bucket_key).name
            logging.info(f"Staging {zarr_path} → {local_store}")
			# pick creds-aware filesystem
            if aws_creds:
                fs2 = s3fs.S3FileSystem(
                    anon=False,
                    key=aws_creds["access_key"],
                    secret=aws_creds["secret_access_key"],
                )
            else:
                fs2 = fs  # module‐level s3fs.S3FileSystem()

            fs2.get(bucket_key, str(local_store), recursive=True)
            logging.info(f"Staged to {local_store}")

            # Now force and re-point `zarr_path` to point at the local copy:
            zarr_path = str(local_store)

        # ─────────────────────────────────────────────────────────────────
        # 2) LOAD “CLIMATOLOGY” ARRAYS FOR NDVI Z‐SCORING (IMMEDIATELY)
        # ─────────────────────────────────────────────────────────────────
        self.seasonal_ndvi = (
            open_dataarray(seasonal_ndvi_path, aws_creds)
            if seasonal_ndvi_path
            else None
        )
        self.monthly_mu    = (
            open_dataarray(monthly_mu_path, aws_creds)
            if monthly_mu_path
            else None
        )
        self.monthly_sigma = (
            open_dataarray(monthly_sigma_path, aws_creds)
            if monthly_sigma_path
            else None
        )

        # ─────────────────────────────────────────────────────────────────
        # 3) DO A ONE‐TIME “METADATA OPEN” TO EXTRACT DIMENSIONS & TIME
        # ─────────────────────────────────────────────────────────────────
        # Use a temporary xarray.Dataset just for shape/time info. Then close it.
        ds_tmp = open_zarr(zarr_path, consolidated=True, aws_creds=aws_creds)

        # (a) Identify which variables are 3D (time, lat, lon) among `bands` or all data_vars
        candidate = bands or list(ds_tmp.data_vars)
        self.bands = [
            b for b in candidate
            if set(ds_tmp[b].dims) == {
                next(d for d in ds_tmp.dims if "time" in d.lower() or "date" in d.lower()),
                next(d for d in ds_tmp.dims if "lat"  in d.lower() or "y"    in d.lower()),
                next(d for d in ds_tmp.dims if "lon"  in d.lower() or "x"    in d.lower())
            }
        ]
        if not self.bands:
            ds_tmp.close()
            raise ValueError(f"No 3D bands among candidate list {candidate}")

        for b in self.bands:
            logging.info(f"band: {b}")

        # (b) Record the dimension names
        D            = list(ds_tmp.dims)
        self.time_dim = next(d for d in D if "time" in d.lower() or "date" in d.lower())
        self.lat_dim  = next(d for d in D if "lat"  in d.lower() or "y"    in d.lower())
        self.lon_dim  = next(d for d in D if "lon"  in d.lower() or "x"    in d.lower())

        # (c) Extract the time array, convert to numpy datetimes, compute max Δt
        self.time  = ds_tmp[self.time_dim].values  # np.ndarray of dtype='datetime64'
        all_dts    = np.diff(self.time) / np.timedelta64(1, 'D')
        self.max_dt = float(np.max(all_dts))

        # (d) Compute the (H, W) shape and sliding‐window start indices
        H, W           = ds_tmp.sizes[self.lat_dim], ds_tmp.sizes[self.lon_dim]
        self.patch_size = patch_size
        self.stride     = stride or patch_size

        def make_starts(full: int, patch: int, stride: int):
            starts = list(range(0, full - patch + 1, stride))
            if starts[-1] + patch < full:
                starts.append(full - patch)
            return starts

        self.row_starts = make_starts(H, patch_size, self.stride)
        self.col_starts = make_starts(W, patch_size, self.stride)
        self.rows       = len(self.row_starts)
        self.cols       = len(self.col_starts)

        # (e) Build the time‐pair indices (list of (i, j) tuples)
        if time_indices is None:
            self.time_indices = [(i, i+1) for i in range(len(self.time)-1)]
        else:
            self.time_indices = time_indices

        self.total_tiles = len(self.time_indices) * self.rows * self.cols
        ds_tmp.close()

        # ─────────────────────────────────────────────────────────────────
        # 4) LOAD “MIN/MAX” FOR EACH BAND (GLOBAL OR PER‐GEOHASH)
        # ─────────────────────────────────────────────────────────────────
		# If both global_min_vals and global_max_vals are supplied, use them directly.
        if (global_min_vals is not None) and (global_max_vals is not None):
			# We assume that the caller built these arrays in exactly the same order as self.bands.
            if len(global_min_vals) != len(self.bands) or len(global_max_vals) != len(self.bands):
                raise ValueError(
                    f"Length of global_min_vals ({len(global_min_vals)}) does not match number of bands ({len(self.bands)})"
                )
			# Turn them into Python floats and store in dicts so that __getitem__ can reference by band name:
            self.band_min = { b: float(global_min_vals[i]) for i,b in enumerate(self.bands) }
            self.band_max = { b: float(global_max_vals[i]) for i,b in enumerate(self.bands) }
        else:
            # Fall back to reading per‐geohash band_stats.json (unchanged logic).
            # We know seasonal_ndvi_path exists if no global_min_vals/global_max_vals.
            if seasonal_ndvi_path is None:
                raise RuntimeError("Need either global_min_vals/global_max_vals or a valid seasonal_ndvi_path")
            if seasonal_ndvi_path.startswith("s3://"):
				# e.g. "s3://bucket/.../9vgm0_climatology/seasonal_ndvi.nc"
                stats_uri = seasonal_ndvi_path.rsplit("/", 1)[0] + "/band_stats.json"
            else:
                stats_path = Path(seasonal_ndvi_path).parent / "band_stats.json"
                stats_uri  = str(stats_path)

            if stats_uri.startswith("s3://"):
                key = stats_uri.removeprefix("s3://")
                so = {}
                if aws_creds:
                    so = {
                        "key":    aws_creds["access_key"],
                        "secret": aws_creds["secret_access_key"],
                        "anon":   False
                    }
                else:
                    so = {"anon": True}

                s3 = s3fs.S3FileSystem(**so)
                if not s3.exists(key):
                    raise FileNotFoundError(f"Could not find band_stats.json on S3 at {stats_uri}")
                with s3.open(key, "r") as f:
                    stats = json.load(f)
            else:
                stats_path = Path(stats_uri)
                if not stats_path.exists():
                    raise FileNotFoundError(f"Could not find band_stats.json locally at {stats_path}")
                with open(stats_path, "r") as f:
                    stats = json.load(f)

			# Build self.band_min / self.band_max from that JSON:
            self.band_min = { b: float(stats[b]["min"]) for b in self.bands }
            self.band_max = { b: float(stats[b]["max"]) for b in self.bands }

        # At this point we have exactly:
        #   self.band_min[b]  = min value of band b
        #   self.band_max[b]  = max value of band b

        # ─────────────────────────────────────────────────────────────────
        # 5) STORE STATE FOR LAZY‐OPEN LATER
        # ─────────────────────────────────────────────────────────────────
        self._zarr_path     = zarr_path
        self._ds            = None   # will hold an xr.Dataset once a worker calls _ensure_open()
        self._fs            = None   # will hold an S3FileSystem once we need it in a worker

    def __len__(self):
        """
        Return the total number of (time-pair, spatial-tile) samples.

        Returns:
            int: Total number of patches = len(time_indices) * rows * cols.
        """
        return self.total_tiles

    def _ensure_open(self):
        """
        Lazily open the Zarr store inside the worker process.

        On first call, constructs an S3FileSystem if needed, opens the Zarr via
        `xr.open_zarr`, and stores the Dataset in `self._ds` for subsequent access.
        Logs the worker ID if within a DataLoader worker.
        """
        if self._ds is None:
            # (a) Create the correct filesystem if loading from S3
            if self._zarr_path.startswith("s3://"):
                if self.aws_creds:
                    self._fs = s3fs.S3FileSystem(
                        anon=False,
                        key=self.aws_creds["access_key"],
                        secret=self.aws_creds["secret_access_key"],
                    )
                else:
                    self._fs = s3fs.S3FileSystem(anon=True)
                mapper = self._fs.get_mapper(self._zarr_path)
                self._ds = xr.open_zarr(mapper, consolidated=True)
            else:
                # Local filesystem path
                self._ds = xr.open_zarr(self._zarr_path, consolidated=True)

            # In principle, we could do additional per-worker setup here if needed
            # Only log a worker‐ID if we actually are in a worker process
            worker = get_worker_info()
            if worker is not None:
                logging.info(f"[Worker {worker.id}] Opened Zarr at {self._zarr_path}")

    def __getitem__(self, idx: int):
        """
        Retrieve a single change-pair sample.

        1. Map `idx` to (pair_idx, tile_idx), derive t0, t1, r0:c0 coordinates.
        2. Read raw bands at t0 and t1 into NumPy arrays.
        3. Compute metadata layers: Δt, DOY/365, seasonal NDVI, z-scored NDVI.
        4. Min-max normalize non-aspect bands and convert "aspect" to sin/cos channels.
        5. Concatenate raw + metadata channels for both t0 and t1.
        6. Stack into shape (2, C_total, H, W), replace NaNs/Infs.
        7. Convert to `torch.Tensor` and return along with scaled Δt tensor.

        Args:
            idx (int): Global sample index in [0, total_tiles).

        Returns:
            tuple:
              - `torch.Tensor` of shape (2, C_total, H, W) with dtype float32.
              - `torch.Tensor` scalar Δt_days / max_dt, dtype float32.
        """
        # Ensure the xarray.Dataset is opened in the worker:
        self._ensure_open()
        # if idx % 37:
        #     logging.info(f"[Dataset] __getitem__ called with idx={idx}")

        # ─────────────────────────────────────────────────────────────────
        # (1) DETERMINE WHICH TIME‐PAIR AND WHICH 2D TILE THIS IDX CORRESPONDS TO
        # ─────────────────────────────────────────────────────────────────
        tp_size  = self.rows * self.cols
        pair_idx = idx // tp_size
        tile_idx = idx % tp_size
        t0, t1   = self.time_indices[pair_idx]
        r_idx    = tile_idx // self.cols
        c_idx    = tile_idx % self.cols
        r0, c0   = self.row_starts[r_idx], self.col_starts[c_idx]
        r1, c1   = r0 + self.patch_size, c0 + self.patch_size

        # ─────────────────────────────────────────────────────────────────
        # (2) EXTRACT RAW BAND PATCHES FOR t0 AND t1 (shape=(len(bands), H, W))
        # ─────────────────────────────────────────────────────────────────
        patches_t0 = []
        patches_t1 = []
        for b in self.bands:
            arr = self._ds[b]
            sel0 = arr.isel({
                self.time_dim: t0,
                self.lat_dim: slice(r0, r1),
                self.lon_dim: slice(c0, c1)
            })
            sel1 = arr.isel({
                self.time_dim: t1,
                self.lat_dim: slice(r0, r1),
                self.lon_dim: slice(c0, c1)
            })
            patches_t0.append(sel0.values)
            patches_t1.append(sel1.values)

		# Convert to a NumPy array (float32) of shape (C, H, W)
        raw_t0 = np.stack(patches_t0, axis=0).astype(np.float32)
        raw_t1 = np.stack(patches_t1, axis=0).astype(np.float32)

        # ─────────────────────────────────────────────────────────────────
        # (3) COMPUTE Δ‐TIME, DOY, MONTH
        # ─────────────────────────────────────────────────────────────────
        dt_days = (self.time[t1] - self.time[t0]) / np.timedelta64(1, 'D')
        doy     = pd.Timestamp(self.time[t0]).dayofyear
        month   = pd.Timestamp(self.time[t0]).month

        # ─────────────────────────────────────────────────────────────────
        # (4) BUILD METADATA LAYERS FOR t1 (Δt, DOY, SEASONAL, Z_NDVI)
        # ─────────────────────────────────────────────────────────────────
        delta_layer = np.full((1, self.patch_size, self.patch_size),
                              dt_days, dtype=np.float32)
        doy_layer   = np.full((1, self.patch_size, self.patch_size),
                              doy / 365.0, dtype=np.float32)

        # ——— seasonal climatology slice ———
        if self.seasonal_ndvi is not None:
            clim = self.seasonal_ndvi.sel(doy=doy, method='nearest')
			# 2) build spatial selector using the actual dim names
            sel_kwargs = {
                self.lat_dim: slice(self._ds[self.lat_dim][r0], self._ds[self.lat_dim][r1-1]),
                self.lon_dim: slice(self._ds[self.lon_dim][c0], self._ds[self.lon_dim][c1-1]),
            }
            seasonal_layer = clim.sel(**sel_kwargs).values[None]
        else:
            seasonal_layer = np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)

        # ——— z‐score NDVI using monthly mean & std ———
        if self.monthly_mu is not None and self.monthly_sigma is not None:
			# 1) pick nearest‐month slices
            mu_base    = self.monthly_mu.sel(month=month, method='nearest')
            sigma_base = self.monthly_sigma.sel(month=month, method='nearest')
			# 2) same spatial selector
            sel_kwargs = {
                self.lat_dim: slice(self._ds[self.lat_dim][r0], self._ds[self.lat_dim][r1-1]),
                self.lon_dim: slice(self._ds[self.lon_dim][c0], self._ds[self.lon_dim][c1-1]),
            }
            mu    = mu_base.sel(**sel_kwargs).values
            sigma = sigma_base.sel(**sel_kwargs).values
            raw_ndvi = patches_t0[self.bands.index('NDVI')]
			# 3) compute z‐score patch
            z_ndvi   = (raw_ndvi - mu) / (sigma + 1e-6)
            z_ndvi_layer = z_ndvi[None]
        else:
            z_ndvi_layer = np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)

        # ─────────────────────────────────────────────────────────────────
        # (5) MIN–MAX NORMALIZE ALL “NON‐ASPECT” BANDS; REPLACE “ASPECT” WITH SIN/COS
        # ─────────────────────────────────────────────────────────────────
        normed_bands_t0: list[np.ndarray] = []
        normed_bands_t1: list[np.ndarray] = []

        for i, b in enumerate(self.bands):
            if b.lower() == "aspect":
				# Convert degrees → radians, then sin & cos. Both are in [-1,1].
                theta0 = raw_t0[i] * (np.pi / 180.0)
                sin0   = np.sin(theta0).astype(np.float32)
                cos0   = np.cos(theta0).astype(np.float32)

                theta1 = raw_t1[i] * (np.pi / 180.0)
                sin1   = np.sin(theta1).astype(np.float32)
                cos1   = np.cos(theta1).astype(np.float32)

                normed_bands_t0.append(sin0)     # new channel: aspect_sin
                normed_bands_t0.append(cos0)     # new channel: aspect_cos
                normed_bands_t1.append(sin1)
                normed_bands_t1.append(cos1)
            else:
                mn  = self.band_min[b]
                mx  = self.band_max[b]
                rng = mx - mn
                if rng < 1e-6:
                    rng = 1.0
				
				# Min-max scaling → [0,1]
                val0 = (raw_t0[i] - mn) / rng
                val1 = (raw_t1[i] - mn) / rng

                val0 = np.clip(val0, 0.0, 1.0)
                val1 = np.clip(val1, 0.0, 1.0)

                normed_bands_t0.append(val0.astype(np.float32))
                normed_bands_t1.append(val1.astype(np.float32))

        # Stack back into arrays of shape ((C + 1), H, W) for t0 and t1:
        #   note: “C + 1” because aspect→(sin,cos) replaced one channel with two.
        raw_t0_norm = np.stack(normed_bands_t0, axis=0)
        raw_t1_norm = np.stack(normed_bands_t1, axis=0)

        # ─────────────────────────────────────────────────────────────────
        # (6) BUILD “ZERO‐METADATA” FOR t0, SO CHANNEL COUNTS MATCH zz
        # ─────────────────────────────────────────────────────────────────
        delta0    = np.zeros_like(delta_layer)
        doy0      = np.zeros_like(doy_layer)
        seasonal0 = np.zeros_like(seasonal_layer)
        z_ndvi0   = np.zeros_like(z_ndvi_layer)

        # ─────────────────────────────────────────────────────────────────
        # (7) CONCATENATE “RAW” + “METADATA” FOR EACH TIME STEP zz
        # ─────────────────────────────────────────────────────────────────
        input_t0 = np.concatenate(
            [raw_t0_norm, delta0, doy0, seasonal0, z_ndvi0],
            axis=0
        )
        input_t1 = np.concatenate(
            [raw_t1_norm, delta_layer, doy_layer, seasonal_layer, z_ndvi_layer],
            axis=0
        )

        # ─────────────────────────────────────────────────────────────────
        # (8) STACK INTO (2, C_total, H, W) AND CLEAN NaNs/INFs zz
        # ─────────────────────────────────────────────────────────────────
		# replace NaNs and infinities with zero
        pair_np    = np.stack([input_t0, input_t1], axis=0)
        pair_np    = np.nan_to_num(pair_np, nan=0.0, posinf=0.0, neginf=0.0)

        # ─────────────────────────────────────────────────────────────────
        # (9) CONVERT TO TORCH TENSOR AND RETURN (pair, Δt_scaled) zz
        # ─────────────────────────────────────────────────────────────────
        pair_tensor = torch.from_numpy(pair_np).float()
		# scale Δt into [0,1] by dividing by a sensible maximum, max
        dt_tensor   = torch.tensor(dt_days, dtype=torch.float32) / self.max_dt

        return pair_tensor, dt_tensor

"""
inspect_dataset.py

Module for loading and inspecting geospatial Zarr datasets: listing variables,
computing band/time summaries, extracting patches, and generating PDF reports
with RGB, NDVI, and static-band visualizations.
"""

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.ioff()
from matplotlib.backends.backend_pdf import PdfPages
import logging
import pymaap
pymaap.init_general_logger()
import datetime

def load_creds(creds_path: str | None) -> dict | None:
    """
    Load AWS credentials from a JSON file.

    Args:
        creds_path: Path to JSON file containing AWS 'access_key' and 'secret_access_key'.

    Returns:
        A dict with AWS credentials, or None if no path is provided.
    """
    if creds_path:
        return json.load(open(creds_path))
    return None

def list_vars(creds_path: str, 
              bucket: str, 
              folder: str,
              geohash: str):
    """
    Connect to an S3 Zarr store and log dataset dimensions and variable summaries.

    Args:
        creds_path: JSON file with AWS credentials for S3 access.
        bucket: Name of the S3 bucket (no s3:// prefix).
        folder: Top-level folder inside the bucket.
        geohash: 5-character geohash code identifying the dataset.

    Returns:
        None. Outputs logs of dataset dimensions and sample stats for key bands.
    """
    # 1) Load creds
    creds = load_creds(creds_path=creds_path)
    fs = s3fs.S3FileSystem(
        anon=False,
        key=creds["access_key"],
        secret=creds["secret_access_key"],
    )

    # 2) Point to your Zarr store
    path = f"{bucket}/{folder}/{geohash}/{geohash}_zarr/"
    mapper = fs.get_mapper(path)

    # 3) Open it
    ds = xr.open_zarr(mapper, consolidated=True)

    # 4) Inspect
    logging.info(f"Dims:       {ds.dims}")
    logging.info(f"Data Vars:  {list(ds.data_vars)}")
    for v in ("elevation","slope","aspect"):
        logging.info(f"  • {v:9s} present? {v in ds.data_vars}")
        if v in ds.data_vars:
            arr = ds[v].isel(time=0).values   # take the first time slice
            logging.info(f"     shape {arr.shape},  min/max = {np.nanmin(arr)!r}/{np.nanmax(arr)!r}")

def open_zarr_from_s3(bucket: str,
                      folder: str,
                      geohash: str,
                      creds: dict | None,
                      consolidated: bool = True) -> xr.Dataset:
    """
    Open a Zarr dataset stored in S3 as an xarray.Dataset.

    Args:
        bucket: S3 bucket name (no prefix).
        folder: Folder path under the bucket.
        geohash: 5-character geohash for dataset location.
        creds: AWS credentials dict or None for anonymous access.
        consolidated: Whether to use consolidated metadata.

    Returns:
        An xarray.Dataset loaded from the S3 Zarr store.
    """
    s3_path = f"{bucket}/{folder}/{geohash}/{geohash}_zarr/"
    opts = {}
    if creds:
        opts = {'key': creds['access_key'], 'secret': creds['secret_access_key']}
    fs = s3fs.S3FileSystem(anon=not bool(creds), **opts)
    mapper = fs.get_mapper(s3_path)
    return xr.open_zarr(mapper, consolidated=consolidated)

def compute_band_summary(ds: xr.Dataset) -> pd.DataFrame:
    """
    Compute min, max, mean, std, and median for each data variable over space and time.

    Args:
        ds: xarray.Dataset with one or more data_vars.

    Returns:
        A DataFrame indexed by variable name with summary statistics.
    """
    stats = {}
    for var in ds.data_vars:
        arr = ds[var].values
        # flatten for median
        flat = arr.flatten()
        flat_nonan = flat[~np.isnan(flat)]
        median = float(np.nanmedian(flat_nonan)) if flat_nonan.size else np.nan
        flat_c = arr.ravel()
        valid = flat_c[~np.isnan(flat_c)]
        count = int(valid.size)
        stats[var] = {
            "min":    float(np.nanmin(arr)),
            "max":    float(np.nanmax(arr)),
            "mean":   float(np.nanmean(arr)),
            "std":    float(np.nanstd(arr)),
            "median": median,
            "count": "N/A"#count
        }
    return pd.DataFrame.from_dict(stats, orient="index")

def compute_time_summary(time_da: xr.DataArray) -> pd.Series:
    """
    Compute simple summary for a datetime coordinate array.

    Args:
        time_da: xarray.DataArray of datetime values.

    Returns:
        A Series with min, max, and count of dates as strings.
    """
    # turn into a flat list of date‐objects
    tlist = list(time_da.values)

    # format each element to "YYYY-MM-DD"
    fmt_list = [_fmt_date(t) for t in tlist]

    # min/max on the formatted strings is valid lexically
    tmin = min(fmt_list)
    tmax = max(fmt_list)
    tcount = len(fmt_list)

    return pd.Series({
        "min":    tmin,
        "max":    tmax,
        "mean":   "N/A",
        "std":    "N/A",
        "median": "N/A",
        "count":  tcount
    })

def prepare_pdf(
        save_path: Path, 
        geohash: str, 
        idx0: int,
        idx1: int,
        time0: str | None = None, 
        time1: str | None = None,
        date0 = None,
        date1 = None      
):
    """
    Prepare output directory and PdfPages for report generation.

    Args:
        save_path: Base directory for saving outputs.
        geohash: Geohash identifier.
        idx0, idx1: Integer time indices used for tagging.
        time0, time1: Optional original time strings.
        date0, date1: Human-readable dates for report tag.

    Returns:
        pdf: PdfPages object for appending report pages.
        out_dir: Directory where outputs will be saved.
        pdf_path: Full path to the PDF file.
        base: Base path including geohash and time tags.
        tag: String tag used for naming.
    """
    logging.info('Preparing PDF/report paths')
    # figure out our output directory
    out_dir = None
    use_time = (time0 is not None and time1 is not None)
    if save_path:
        # base = whatever user passed
        base = Path(save_path)
        # optionally nest under geohash and index
        if geohash:
            base = base / f"geohash_{geohash}"
            if use_time:
                # encode both integer indices and dates
                # tag = f"{idx0}_{time0}_to_{idx1}_{time1}"
                tag = f"{idx0}_{date0}_to_{idx1}_{date1}"
                base = base / f"time_{tag}"
            else:
                # just the starting index
                tag = str(idx0)
                base = base / f"index_{tag}"
        # make it
        base.mkdir(parents=True, exist_ok=True)
        out_dir = base
        logging.info(f"Writing all outputs into {out_dir}")
        # create the PDF in that dir
        pdf_path = out_dir / f"report_{tag}.pdf"
        pdf = PdfPages(pdf_path)
    else:
        pdf = None
    return pdf, out_dir, pdf_path, base, tag

def detect_dims_bands(ds: xr.Dataset, show_ndvi: bool) -> tuple[
    list[str], str, str, str, list[str], str, list[str]
]:
    """
    Inspect an xarray Dataset to find its time, latitude, longitude dimensions
    and identify the names of the RGB, NDVI (optional), and static bands.

    Args:
        ds (xr.Dataset): The input dataset containing spatial and temporal dims.
        show_ndvi (bool): Whether to require and return the NDVI band name.

    Returns:
        Dims (list[str]): All dimension names in the dataset.
        tdim (str): Name of the time dimension (e.g. "time" or "date").
        lat (str): Name of the latitude dimension (e.g. "lat" or "y").
        lon (str): Name of the longitude dimension (e.g. "lon" or "x").
        rgb_bands (list[str]): Names of the R, G, B bands in order.
        ndvi_band (str): Name of the NDVI band (only if show_ndvi=True).
        static_bands (list[str]): Names of the static bands ["elevation","slope","aspect"].
    """
    logging.info('Setting up bands')
    Dims = list(ds.dims)
    logging.info(f"Dataset dims: {Dims}")
    # time can be "time" or "date"
    tdim = next(d for d in Dims if 'time' in d.lower() or 'date' in d.lower())
    # lat may be named "lat" or simply "y"
    lat  = next(d for d in Dims if 'lat' in d.lower() or d.lower() == 'y')
    # lon may be named "lon" or simply "x"
    lon  = next(d for d in Dims if 'lon' in d.lower() or d.lower() == 'x')

    var_map = {v.lower(): v for v in ds.data_vars}
    # RGB
    desired_rgb = ['r','g','b']
    rgb_bands = []
    for lr in desired_rgb:
        if lr in var_map:
            rgb_bands.append(var_map[lr])
        else:
            raise ValueError(f"Band '{lr}' missing; available: {list(ds.data_vars)}")
    # NDVI
    if show_ndvi:
        if 'ndvi' in var_map:
            ndvi_band = var_map['ndvi']
        else:
            raise ValueError(f"Cannot find 'ndvi' band; available: {list(ds.data_vars)}")
    # static bands
    desired_static = ['elevation','slope','aspect']
    static_bands = []
    for ls in desired_static:
        if ls in var_map:
            static_bands.append(var_map[ls])
        else:
            raise ValueError(f"Cannot find static band '{ls}'; available: {list(ds.data_vars)}")
    return Dims, tdim, lat, lon, rgb_bands, ndvi_band, static_bands

def make_starts(full: int, patch: int, stride: int) -> list[int]:
    """
    Compute the list of starting indices along one dimension to cover
    it fully with patches of a given size and stride.

    Args:
        full (int): Full size of the dimension.
        patch (int): Patch size.
        stride (int): Stride between patch starts.

    Returns:
        List of start positions so that the last patch touches the end.
    """

    starts = list(range(0, full - patch + 1, stride))
    if starts[-1] + patch < full:
        starts.append(full - patch)
    return starts

def resolve_time_indices(
    time0: str | int | None,
    time1: str | int | None,
    tdim: str,
    ds: xr.Dataset,
    tt0: int,
    tt1: int
) -> tuple[int, int]:
    """
    Snap ISO dates or date objects to nearest index,
    then clamp to ensure t0 < t1 and within bounds.

    Args:
        time0, time1: Strings ("YYYY-MM-DD") or datetime.date or ints.
        tdim (str): Time dimension name.
        ds (xr.Dataset): Dataset containing that time axis.
        tt0, tt1 (int): Fallback indices.

    Returns:
        (t0, t1): Integer indices satisfying 0 ≤ t0 ≤ T−2 and t0+1 ≤ t1 ≤ T−1.
    """
    # how many steps we have
    T = ds[tdim].size

    # 1) if they really passed two dates, snap those to nearest indices
    if isinstance(time0, (str, datetime.date)) and isinstance(time1, (str, datetime.date)):
        # build a day‐resolution array
        arr_days = ds[tdim].values.astype("datetime64[D]")
        def to_day64(d):
            if isinstance(d, str):
                d = datetime.date.fromisoformat(d)
            return np.datetime64(d, "D")
        day0 = to_day64(time0)
        day1 = to_day64(time1)
        i0 = int((np.abs(arr_days - day0)).argmin())
        i1 = int((np.abs(arr_days - day1)).argmin())
        t0, t1 = i0, i1
    else:
        # otherwise use whatever fallback ints you already computed
        t0, t1 = tt0, tt1

    # 2) now clamp into a strictly forward window:
    #    0 <= t0 <= T-2, and t1 in [t0+1, T-1]
    t0 = max(0, min(int(t0), T - 2))
    t1 = max(t0 + 1, min(int(t1), T - 1))
    return t0, t1

def get_patch(
    ds: xr.Dataset,
    var: str,
    r0: int, r1: int,
    c0: int, c1: int,
    time_dim: str = "time",
    t: int | None = None,
    lat: str = "lat",
    lon: str = "lon",
) -> np.ndarray:
    """
    Extract a 2D (H×W) or 3D (T×H×W) numpy array of a single variable from
    an xarray Dataset given spatial and optional temporal slices.

    Args:
        ds (xr.Dataset): Source dataset.
        var (str): Data variable name.
        r0, r1 (int): Row slice indices.
        c0, c1 (int): Column slice indices.
        time_dim (str): Name of the time dimension.
        t (int|None): Single time index (if provided).
        lat (str), lon (str): Names of spatial dims.

    Returns:
        np.ndarray: The sliced array.
    """
    indexers: dict[str, slice | int] = {
        lat: slice(r0, r1),
        lon: slice(c0, c1),
    }
    if t is not None:
        indexers[time_dim] = t
    return ds[var].isel(indexers).values

def _fmt_date(t) -> str:
    """
    Normalize various date types (numpy datetime64, Python datetime,
    cftime) to "YYYY-MM-DD" string.

    Args:
        t: Date-like object.

    Returns:
        Formatted date string.
    """

    try:
        return pd.to_datetime(t).strftime("%Y-%m-%d")
    except Exception:
        pass
    try:
        return t.strftime("%Y-%m-%d")
    except Exception:
        pass
    # fallback in case t has year/month/day attributes
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}"

def time_indices_to_dates(
    ds: xr.Dataset,
    idx0: int,
    idx1: int,
) -> tuple[str, str]:
    """
    Reverse mapping of two integer indices back to ISO date strings
    using the dataset's 'time' coordinate.

    Args:
        ds (xr.Dataset): Dataset containing a 'time' coordinate.
        idx0, idx1 (int): Indices along that coordinate.

    Returns:
        (date0, date1): ISO date strings for each index.
    """
    logging.info("Calculating dates from time indices...")

    # pull out the raw time objects
    t0 = ds["time"].isel(time=idx0).item()
    t1 = ds["time"].isel(time=idx1).item()

    date0 = _fmt_date(t0)
    date1 = _fmt_date(t1)

    print("date0:", date0, "\ndate1:", date1)
    return date0, date1

def rgb_before_after(
    ds: xr.Dataset,
    rgb_bands: list[str],
    r0: int, r1: int,
    c0: int, c1: int,
    tt0: int, tt1: int,
    tdim: str,
    lat_dim: str, lon_dim: str,
    idx0: int, idx1: int, total: int,
    patch_loc: int,
    out_dir: Path, pdf: PdfPages,
    tag: str,
    geohash: str,
    tiles_per_slice: int
):
    """
    Generate and save a true-color comparison of the full scene and a single patch
    at two time points.

    Creates a 2×2 figure:
      - Top row: full-scene RGB at t₀ and t₁, with a red box showing the patch
      - Bottom row: zoomed-in RGB patch at t₀ and t₁

    Args:
        ds (xr.Dataset): Dataset containing the bands and coordinates.
        rgb_bands (list[str]): Names of the R, G, B variables.
        r0, r1 (int): Row slice bounds of the patch.
        c0, c1 (int): Column slice bounds of the patch.
        tt0, tt1 (int): Time indices for before/after comparison.
        tdim (str): Name of the time dimension.
        lat_dim, lon_dim (str): Names of spatial dimensions.
        idx0, idx1 (int): Global indices of the time-patch pair.
        total (int): Total number of such pairs in the dataset.
        patch_loc (int): Linear index of this patch within one time slice.
        out_dir (Path): Directory to save the PNG.
        pdf (PdfPages): Open PDF to which the figure will also be added.
        tag (str): String tag for filenames (e.g. "17_2020-01-01_to_...").
        geohash (str): Geohash code for the region.
        tiles_per_slice (int): Number of patches per time slice.
    """
    # --- human‐readable dates ---
    tvals = ds[tdim].values
    if tt0 > len(tvals)-1 or tt1 > len(tvals)-1:
        tt0, tt1 = len(tvals)-2, len(tvals)-1
    d0 = _fmt_date(tvals[tt0])
    d1 = _fmt_date(tvals[tt1])
    date0, date1 = time_indices_to_dates(ds, idx0, idx1)

    # --- get geographic coords & extents ---
    lats = ds[lat_dim].values
    lons = ds[lon_dim].values
    # imshow extent is [xmin, xmax, ymin, ymax]
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    # --- helper to stretch for display ---
    def stretch(img):
        lo, hi = np.percentile(img, (1,99))
        return np.clip((img - lo)/(hi - lo + 1e-6), 0,1)

    # --- full‐frame RGB at t0/t1 ---
    full0 = np.stack([ds[b].isel({tdim: tt0}).values for b in rgb_bands], axis=0)
    full1 = np.stack([ds[b].isel({tdim: tt1}).values for b in rgb_bands], axis=0)
    rgb_full0 = stretch(np.moveaxis(full0, 0, -1))
    rgb_full1 = stretch(np.moveaxis(full1, 0, -1))

    # --- patch RGB at t0/t1 ---
    p0 = np.stack([
        get_patch(ds, b, r0, r1, c0, c1, time_dim=tdim, t=tt0,
                  lat=lat_dim, lon=lon_dim)
        for b in rgb_bands
    ], axis=0)
    p1 = np.stack([
        get_patch(ds, b, r0, r1, c0, c1, time_dim=tdim, t=tt1,
                  lat=lat_dim, lon=lon_dim)
        for b in rgb_bands
    ], axis=0)
    rgb0 = stretch(np.moveaxis(p0, 0, -1))
    rgb1 = stretch(np.moveaxis(p1, 0, -1))

    # --- compute patch corner coords in lon/lat ---
    lon0, lon1 = float(lons[c0]), float(lons[c1-1])
    lat0, lat1 = float(lats[r1-1]), float(lats[r0])

    # --- 2×2 figure: full / patch before & after ---
    fig, axes = plt.subplots(2, 2, figsize=(12,12), constrained_layout=False)
    fig.subplots_adjust(top=0.88, hspace=0.3)
    fig.suptitle(
        f"True-Color\nGeohash: {geohash}\nIndex: {idx0}/{total}\nPatch Location: {patch_loc+1}/{tiles_per_slice}\nDate: {date0} → {date1}",
        y=1.01,
        fontsize=14
    )

    # Top row: full frames
    for ax, img_full, d in zip(axes[0], (rgb_full0, rgb_full1), (d0, d1)):
        ax.imshow(img_full, extent=extent, origin='upper')
        ax.set_title(f"Full frame: {d}", fontsize=13)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.5)
        # draw patch rectangle
        width  = lon1 - lon0
        height = lat1 - lat0
        rect = Rectangle(
            (lon0, lat0), width, height,
            edgecolor="red", facecolor="none", linewidth=2
        )
        ax.add_patch(rect)

    # Bottom row: patches (with their own local extents)
    # compute local extent for patch axes so ticks still meaningful
    patch_extent = [lon0, lon1, lat0, lat1]
    for ax, img_patch, d in zip(axes[1], (rgb0, rgb1), (d0, d1)):
        ax.imshow(img_patch, extent=patch_extent, origin='upper')
        ax.set_title(f"Patch: {d}", fontsize=13)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.5)

    # --- save & close ---
    if out_dir:
        fname = out_dir / f"rgb_{tag}.png"
        fig.savefig(fname, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def ndvi_before_after(
    ds: xr.Dataset,
    ndvi_band: str,
    r0: int, r1: int,
    c0: int, c1: int,
    tt0: int, tt1: int,
    tdim: str,
    lat_dim: str, lon_dim: str,
    idx0: int, idx1: int, total: int,
    patch_loc: int,
    out_dir: Path, pdf: PdfPages,
    tag: str,
    geohash: str,
    tiles_per_slice: int,
    cmap: str = "viridis"
):
    """
    Generate and save NDVI panels for a full scene and a single patch at two times.

    Creates a 2×3 figure:
      - Top row: full-scene NDVI at t₀, t₁, and their difference (ΔNDVI)
      - Bottom row: patch NDVI at t₀, t₁, and ΔNDVI

    Args:
        ds (xr.Dataset): Dataset containing the NDVI variable.
        ndvi_band (str): Name of the NDVI variable.
        r0, r1 (int): Row slice bounds of the patch.
        c0, c1 (int): Column slice bounds of the patch.
        tt0, tt1 (int): Time indices for comparison.
        tdim (str): Name of the time dimension.
        lat_dim, lon_dim (str): Names of spatial dimensions.
        idx0, idx1 (int): Global indices of the time-patch pair.
        total (int): Total number of pairs.
        patch_loc (int): Linear index of this patch.
        out_dir (Path): Directory to save the PNG.
        pdf (PdfPages): Open PDF for adding the figure.
        tag (str): Filename tag.
        geohash (str): Geohash for the region.
        tiles_per_slice (int): Number of patches per time slice.
        cmap (str): Matplotlib colormap for NDVI.
    """
    # --- human‐readable dates & clamp ---
    tvals = ds[tdim].values
    tt0 = min(tt0, len(tvals)-1)
    tt1 = min(tt1, len(tvals)-1)
    date0, date1 = time_indices_to_dates(ds, idx0, idx1)

    # --- spatial coords & extents ---
    lats = ds[lat_dim].values
    lons = ds[lon_dim].values
    full_extent = [
        float(lons.min()), float(lons.max()),
        float(lats.min()), float(lats.max())
    ]
    # corner coords of patch
    lon0, lon1 = float(lons[c0]), float(lons[c1-1])
    lat0, lat1 = float(lats[r1-1]), float(lats[r0])
    patch_extent = [lon0, lon1, lat0, lat1]

    # --- pull data ---
    full0 = ds[ndvi_band].isel({tdim: tt0}).values
    full1 = ds[ndvi_band].isel({tdim: tt1}).values
    patch0 = get_patch(ds, ndvi_band, r0, r1, c0, c1,
                       time_dim=tdim, t=tt0, lat=lat_dim, lon=lon_dim)
    patch1 = get_patch(ds, ndvi_band, r0, r1, c0, c1,
                       time_dim=tdim, t=tt1, lat=lat_dim, lon=lon_dim)

    diff_full  = full1 - full0
    diff_patch = patch1 - patch0

    # --- figure & layout ---
    fig, axes = plt.subplots(2, 3, figsize=(15,10), constrained_layout=False)
    fig.subplots_adjust(top=0.88, hspace=0.3, wspace=0.2)
    fig.suptitle(
        f"NDVI\nGeohash: {geohash}\nIndex: {idx0}/{total}\nPatch Location: {patch_loc+1}/{tiles_per_slice}\nDate: {date0} → {date1}",
        y=1.03,
        fontsize=14
    )

    # --- top row: full‐frame t0, t1, Δ ---
    titles_full = ("Full NDVI t₀", "Full NDVI t₁", "ΔNDVI Full")
    arrays_full = (full0, full1, diff_full)
    vmin, vmax = -1, 1
    for ax, arr, title in zip(axes[0], arrays_full, titles_full):
        im = ax.imshow(arr, extent=full_extent, origin="upper",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        # red box on the two full frames only
        if "Full" in title:
            rect = Rectangle(
                (lon0, lat0),
                lon1 - lon0, lat1 - lat0,
                edgecolor="red", facecolor="none", linewidth=2
            )
            ax.add_patch(rect)
        fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")

    # --- bottom row: patch t0, t1, Δ ---
    titles_patch = ("NDVI Patch t₀", "NDVI Patch t₁", "ΔNDVI Patch")
    arrays_patch = (patch0, patch1, diff_patch)
    for ax, arr, title in zip(axes[1], arrays_patch, titles_patch):
        im = ax.imshow(arr, extent=patch_extent, origin="upper",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")

    # --- save & close ---
    if out_dir:
        fname = out_dir / f"ndvi_{tag}.png"
        fig.savefig(fname, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def static_band_maps(
    ds: xr.Dataset,
    static_bands: list[str],   # e.g. ["elevation","slope","aspect"]
    r0: int, r1: int,
    c0: int, c1: int,
    tt0: int,
    lat_dim: str,
    lon_dim: str,
    idx0: int,
    total: int,
    patch_loc: int,
    out_dir: Path,
    pdf: PdfPages,
    tag: str,
    geohash: str,
    tiles_per_slice: int,
    date0: str, date1: str,
    time_dim: str = 'time'
):
    """
    Generate and save static-layer maps (elevation, slope, aspect) before/after.

    Creates a 2×3 figure:
      - Row 1: full-scene elevation, slope, aspect
      - Row 2: patch elevation, slope, aspect

    Args:
        ds (xr.Dataset): Dataset containing the static bands.
        static_bands (list[str]): Names in order ["elevation","slope","aspect"].
        r0, r1 (int): Row slice bounds of the patch.
        c0, c1 (int): Column slice bounds of the patch.
        tt0 (int): Time index to use for static layers.
        lat_dim, lon_dim (str): Names of spatial dimensions.
        idx0 (int): Global index of the time-patch pair.
        total (int): Total number of pairs.
        patch_loc (int): Linear index of this patch.
        out_dir (Path): Directory to save the PNG.
        pdf (PdfPages): Open PDF for adding the figure.
        tag (str): Filename tag.
        geohash (str): Geohash code.
        tiles_per_slice (int): Patches per time slice.
        date0, date1 (str): ISO dates for t₀ and t₁ (for title).
        time_dim (str): Name of the time dimension.
    """
    # 1) extents
    lats = ds[lat_dim].values
    lons = ds[lon_dim].values
    full_extent = [float(lons.min()), float(lons.max()),
                   float(lats.min()), float(lats.max())]
    lon0, lon1 = float(lons[c0]), float(lons[c1-1])
    lat0, lat1 = float(lats[r1-1]), float(lats[r0])
    patch_extent = [lon0, lon1, lat0, lat1]

    # 2) prefs
    prefs = [
        (static_bands[0], "terrain", "Elevation (m)"),
        (static_bands[1], "viridis", "Slope (°)"),
        (static_bands[2], "viridis", "Aspect (°)")
    ]

    # 3) figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=False)
    fig.subplots_adjust(top=0.88, hspace=0.3, wspace=0.2)
    fig.suptitle(
        f"Static Maps\nGeohash: {geohash}\nIndex: {idx0}/{total}\nPatch Location: {patch_loc+1}/{tiles_per_slice}\nDate: {date0} → {date1}",
        y=1.06, fontsize=18
    )

    # 4) loop
    for col, (var, cmap, label) in enumerate(prefs):
        # slice down to 2D
        if time_dim in ds[var].dims:
            img_full = ds[var].isel({time_dim: tt0})
        else:
            img_full = ds[var]
        arr_full  = img_full.values
        arr_patch = img_full.isel({lat_dim: slice(r0, r1),
                                   lon_dim: slice(c0, c1)}).values

        # full
        ax_full = axes[0, col]
        im0 = ax_full.imshow(arr_full,
                             extent=full_extent,
                             origin="upper",
                             cmap=cmap)
        ax_full.set_title(f"Full {label}", fontsize=16, pad=10)
        ax_full.set_xlabel("Longitude"); ax_full.set_ylabel("Latitude")
        ax_full.grid(True, linestyle="--", alpha=0.4)
        ax_full.add_patch(Rectangle(
            (lon0, lat0),
            lon1 - lon0, lat1 - lat0,
            edgecolor="red", facecolor="none", linewidth=3
        ))
        fig.colorbar(im0, ax=ax_full, shrink=0.7, label=label)

        # patch
        ax_patch = axes[1, col]
        im1 = ax_patch.imshow(arr_patch,
                              extent=patch_extent,
                              origin="upper",
                              cmap=cmap)
        ax_patch.set_title(f"Patch {label}", fontsize=16, pad=10)
        ax_patch.set_xlabel("Longitude"); ax_patch.set_ylabel("Latitude")
        ax_patch.grid(True, linestyle="--", alpha=0.4)
        fig.colorbar(im1, ax=ax_patch, shrink=0.7, label=label)

    # 5) save & close
    if out_dir:
        png_path = out_dir / f"static_{tag}.png"
        fig.savefig(png_path, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def add_dataset_summary_page(
    ds: xr.Dataset,
    time_dim: str,
    pdf: PdfPages,
    geohash: str,
    out_dir: Path,
    fig_width: float = 8,
    row_height: float = 0.4,
    font_size: int = 10
):
    """
    Append a summary table of dataset statistics as the final page in a PDF.

    This function computes:
      1. Time-range summary (min, max, count) for the time coordinate.
      2. Per-band summary (min, max, mean, std, median) for each data variable.
    It then renders these statistics in a tabular figure sized to fit
    the number of rows and adds it to the open PDF.

    Args:
        ds (xr.Dataset): The dataset to summarize.
        time_dim (str): Name of the time dimension in `ds`.
        pdf (PdfPages): An open PdfPages object to which the summary page will be added.
        geohash (str): Geohash identifier for labeling/logging.
        out_dir (Path): Directory where a CSV copy of the summary table will be saved.
        fig_width (float): Width of the summary-figure in inches.
        row_height (float): Height per row in the figure.
        font_size (int): Base font size for the table cells.
    """
    logging.info("Adding dataset summary page to PDF")

    # --- 1) compute time summary and band summary ---
    time_ser   = compute_time_summary(ds[time_dim]) # e.g. {"min":..., "max":..., "count":...}
    band_df    = compute_band_summary(ds).round(3)  # DataFrame indexed by band, with min/max/mean/std/median
    # stick the time row on top
    summary_df = pd.concat(
        [pd.DataFrame([time_ser], index=["time"]), band_df],
        axis=0,
        sort=False
    )
    # reorder rows into logical sequence
    desired_order = [
        "time",
        "R", "G", "B",
        "NDVI",
        "elevation",
        "slope",
        "aspect",
        "aspect_mask",
        "delta_t_norm",
        "doy_norm"
        # other channels added here to be up front...
    ]
    # keep only the ones we actually have
    front = [r for r in desired_order if r in summary_df.index]
    # collect any bands not mentioned above
    rest  = [r for r in summary_df.index if r not in front]
    # reindex
    summary_df = summary_df.loc[ front + rest ]

    # save summary_df as CSV
    csv_out_dir = out_dir / "summary_table.csv"
    try:
        summary_df.to_csv(csv_out_dir)
        logging.info(f"Summary table saved to {csv_out_dir}")
    except Exception as e:
        logging.error(f"Failed to save summary table to CSV: {e}")

    # --- 2) make a figure sized to fit all rows ---
    n_rows = len(summary_df)
    fig_height = 0.5 + row_height * n_rows
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.axis("off")

    # --- 3) render the table ---
    table = ax.table(
        cellText=summary_df.values,
        rowLabels=summary_df.index,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.5)

    # --- 4) add to PDF ---
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def inspect_tile_from_ds(
    ds: xr.Dataset,
    idx0: int,
    idx1: int,
    patch_loc: int,
    tiles_per_slice: int,
    time0: str | None = None,
    time1: str | None = None,
    patch_size: int = 128,
    rgb_bands: list[str] = None,
    static_bands: list[str] = None,
    show_ndvi: bool = True,
    save_path: Path = None,
    geohash: str = None
):
    """
    Generate a suite of before/after visualizations for a single spatiotemporal patch.

    This orchestrator function walks through:
      1. Detecting dimension and band names in the dataset.
      2. Resolving the requested time indices (idx0, idx1) to valid positions.
      3. Computing the patch’s row/column bounds from `patch_loc`.
      4. Producing:
         - True-color RGB before/after images
         - NDVI before/after + difference plots
         - Static-band maps (elevation, slope, aspect)
      5. Saving each figure as PNG in `save_path` and adding them to a PDF report.
      6. Finalizing the PDF with a summary page.

    Args:
        ds (xr.Dataset): Input dataset containing time and spatial dims plus bands.
        idx0 (int): Global index for the “before” time-patch pair.
        idx1 (int): Global index for the “after” time-patch pair.
        patch_loc (int): Linear index of patch within one time slice.
        tiles_per_slice (int): Number of patches per time slice.
        time0 (str | None): Optional ISO date for before; if None, uses idx0.
        time1 (str | None): Optional ISO date for after; if None, uses idx1.
        patch_size (int): Height/width of square patch in pixels.
        rgb_bands (list[str] | None): Names of the R,G,B variables (detected if None).
        static_bands (list[str] | None): Names of elevation, slope, aspect (detected if None).
        show_ndvi (bool): Whether to include NDVI visualizations.
        save_path (Path | None): Base directory for saving PNGs and PDF; if None, no files are written.
        geohash (str | None): Geohash code for logging and filenames.
    """
    # --- detect dims & bands ---
    Dims, tdim, lat, lon, rgb_bands, ndvi_band, static_bands = detect_dims_bands(
        ds=ds, show_ndvi=show_ndvi)
    tt0, tt1 = idx0, idx1

    # --- compute patch row/col from user patch_loc ---
    H, W = ds.sizes[lat], ds.sizes[lon]
    rows = make_starts(H, patch_size, patch_size)
    cols = make_starts(W, patch_size, patch_size)
    R, C = len(rows), len(cols)
    total = len(ds[tdim]) * R * C  # for title/metadata only

    # clamp patch_loc into the valid range
    max_pl = R * C - 1
    pl = max(0, min(patch_loc, max_pl))
    r_idx, c_idx = divmod(pl, C)
    r0, c0 = rows[r_idx], cols[c_idx]
    r1, c1 = r0 + patch_size, c0 + patch_size

    # --- calculate dates (yyyy-mm-dd) from indices
    date0, date1 = time_indices_to_dates(ds, idx0, idx1)

    # --- prepare PDF if needed ---
    pdf, out_dir, pdf_path, base, tag = prepare_pdf(
        save_path=save_path,
        geohash=geohash,
        idx0=idx0,
        idx1=idx1,
        time0=time0,
        time1=time1,
        date0=date0,
        date1=date1)

    # --- find nearest time indices ---
    print(f"time0, time1: {time0}, {type(time0)}, {time1}, {type(time1)}")

    tt0, tt1 = resolve_time_indices(time0, time1, tdim, ds, tt0, tt1)
    print(f"tt0, tt1: {tt0}, {type(tt0)}, {tt1}, {type(tt1)}")

    # --- 1) RGB before/after ---
    rgb_before_after(ds=ds, rgb_bands=rgb_bands, r0=r0, r1=r1, c0=c0, c1=c1, 
                      tt0=tt0, tt1=tt1, tdim=tdim, lat_dim=lat, lon_dim=lon,
                      idx0=idx0, idx1=idx1, total=total, patch_loc=patch_loc,
                      out_dir=out_dir, pdf=pdf, tag=tag, geohash=geohash, 
                      tiles_per_slice=tiles_per_slice)

    # --- 2) NDVI panels ---
    ndvi_before_after(ds=ds, ndvi_band=ndvi_band, r0=r0, r1=r1, c0=c0, c1=c1, 
                       tt0=tt0, tt1=tt1, tdim=tdim, lat_dim=lat, lon_dim=lon,
                       idx0=idx0, idx1=idx1, total=total, patch_loc=patch_loc,
                       out_dir=out_dir, pdf=pdf, tag=tag, geohash=geohash,
                       tiles_per_slice=tiles_per_slice)

    # --- 3) Static-band maps ---
    static_band_maps(ds=ds, static_bands=static_bands, r0=r0, r1=r1, c0=c0, c1=c1, 
                     tt0=tt0, lat_dim=lat, lon_dim=lon, idx0=idx0, 
                     total=total, out_dir=out_dir, pdf=pdf, tag=tag, geohash=geohash, 
                     tiles_per_slice=tiles_per_slice, patch_loc=patch_loc, 
                     date0=date0, date1=date1)

    # --- finalize PDF with a summary page and close it out ---
    if save_path:
        logging.info("Finalizing PDF report")
        add_dataset_summary_page(ds, tdim, pdf, geohash=geohash, out_dir=out_dir)
        pdf.close()
        logging.info(f"PDF report written to {pdf_path}")

def dates_to_time_indices(
    ds: xr.Dataset,
    dates: tuple[str | datetime.date, str | datetime.date],
    time_dim: str = "time",
) -> tuple[int, int]:
    """
    Map two calendar dates to the nearest integer indices along a Dataset’s time axis.

    Given a dataset `ds` with a time coordinate `time_dim`, and a pair of target dates
    (as ISO strings or `datetime.date`), this function:
      1. Coerces both inputs to `numpy.datetime64[D]`.
      2. Finds the index in `ds[time_dim]` whose date is closest to each target.
      3. Returns the two indices as a `(idx0, idx1)` tuple.

    Args:
        ds (xr.Dataset): Dataset containing the time coordinate.
        dates (tuple[str or date, str or date]): Two target dates for lookup.
        time_dim (str): Name of the time coordinate in `ds` (default: "time").

    Returns:
        tuple[int, int]: Indices into `ds[time_dim]` nearest to the two dates.
    """
    # pull out the time values as day‐resolution
    times = ds[time_dim].values.astype("datetime64[D]")
    # normalize input to numpy datetime64[D]
    targets = []
    for d in dates:
        if isinstance(d, str):
            d_obj = pd.to_datetime(d).date()
        elif isinstance(d, datetime.datetime):
            d_obj = d.date()
        else:
            d_obj = d  # assume it's a datetime.date
        targets.append(np.datetime64(d_obj.isoformat(), "D"))

    idxs = []
    for tgt in targets:
        # absolute difference in days
        deltas = np.abs(times - tgt)
        idxs.append(int(deltas.argmin()))

    return idxs[0], idxs[1]

def parse_args():
    """
    Parse command-line arguments for the dataset inspection script.

    Defines a CLI with two mutually exclusive modes:
      - `--index / -i`: a single global patch index to inspect.
      - `--dates / -d`: a pair of dates (ISO strings or "first"/"last").

    Also accepts:
      - S3 bucket & folder (`--bucket`, `--folder`)
      - Geohash key (`--geohash`)
      - Output base path for PNGs/PDF (`--output`)
      - AWS credentials JSON (`--aws-creds`)
      - Patch sizing (`--patch-size`, `--patch-location`)

    Returns:
        argparse.Namespace: Parsed arguments object.
    """
    p = argparse.ArgumentParser(
        description="Inspect RGB, NDVI, static bands, arbitrary time-paris, and produce PDF report"
    )
    p.add_argument("--bucket", "-b", required=True)
    p.add_argument("--folder", "-f", type=str, default="data", required=True)
    p.add_argument("--geohash", "-g", type=str, required=True)
    p.add_argument("--output", "-o", type=Path,
                   help="Base path (no extension) for PNG + PDF output")
    p.add_argument("--aws-creds", "-a", help="JSON file with AWS keys")
    p.add_argument("--patch-size", "-p", type=int, default=128)
    p.add_argument("--patch-location", "-pl", type=int, default=0,
                   help="Index of the patch within a single time slice (row x col ordering, 0=top-left)")
    # mutually exclusive: either index or two dates
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", "-i", type=int, help=(
        "Inspect the patch at this global index, then "
        "inspect the same patch location in the next time slice"
    ))
    group.add_argument("--dates", "-d", nargs=2, type=str, metavar=("DATE0", "DATE1"), help=(
        "Inspect by two dates. Each may be YYYY-MM-DD or the keywords "
        "'first' / 'last' to grab the first/last time slice."
    ))
    return p.parse_args()

def main():
    """
    Entry point for the dataset inspection tool.

    1. Parses CLI arguments.
    2. Loads AWS creds and opens the Zarr store for the given geohash.
    3. Computes the number of spatial patches per time slice.
    4. Resolves the user’s request via `--index` or `--dates` into two time indices.
    5. Clamps indices and patch‐location to valid ranges.
    6. (Optionally) logs diagnostic info.
    7. Calls `inspect_tile_from_ds` to generate and save the visualizations & PDF report.
    """
    args = parse_args()

    # determine patch_loc within EACH slice
    # clamp it so 0 <= patch_loc < (rows * cols)
    # we'll get rows, cols after we open the Zarr, but for now store it
    user_patch_loc = args.patch_location

    if len(str(args.geohash)) != 5:
        logging.error(f"Error: Geohash must be five digits, yours is {len(str(args.geohash))}")
        sys.exit(1)

    # open zarr
    creds = load_creds(args.aws_creds)
    ds = open_zarr_from_s3(
        bucket=args.bucket, 
        folder=args.folder, 
        geohash=args.geohash, 
        creds=creds)
    
    # detect time and spatial dims
    time_dim = next(d for d in ds.dims if "time" in d.lower() or "date" in d.lower())
    lat_dim = next(d for d in ds.dims if "lat" in d.lower() or "y" in d.lower())
    lon_dim = next(d for d in ds.dims if "lon" in d.lower() or "x" in d.lower())
    H, W = ds.sizes[lat_dim], ds.sizes[lon_dim]
    # same sliding logic as in your Dataset
    stride = args.patch_size  # or expose as its own CLI flag if you like
    rows = make_starts(H, args.patch_size, stride)
    cols = make_starts(W, args.patch_size, stride)
    tiles_per_slice = len(rows) * len(cols)
    logging.info(f"Patches per slice (with overlap): {tiles_per_slice}")
    
    # resolve based on --dates or --index
    if args.dates:
        date_str0, date_str1 = args.dates
        def str_to_idx(s: str) -> int:
            s_low = s.lower()
            if s_low == "first":
                return 0
            if s_low == "last":
                return ds.sizes[time_dim] - 1  # time axis length
            # parse an ISO date to nearest index
            return dates_to_time_indices(ds, (s, s))[0]
        idx0 = str_to_idx(date_str0)
        idx1 = str_to_idx(date_str1)
        # ensure chronological order
        if idx1 < idx0:
            idx0, idx1 = idx1, idx0
        time0 = time1 = None
    else:
        # global‐index mode
        idx0 = args.index
        idx1 = idx0 + 1
        time0 = time1 = None
    # clamp t0/t1 against ends
    max_t = ds.sizes[time_dim] - 1
    if idx0 >= max_t:
        idx0, idx1 = max_t - 1, max_t
    # clamp patch_loc
    patch_loc = max(0, min(user_patch_loc, tiles_per_slice - 1))
    # figure out row/col within the slice
    num_cols = len(cols)
    row_idx, col_idx = divmod(patch_loc, num_cols)
    y0, x0 = rows[row_idx], cols[col_idx]
    logging.info(
        f"Clamped patch_loc to {patch_loc} → "
        f"row_idx={row_idx}, col_idx={col_idx} "
        f"(r0={y0}, c0={x0})"
    )

    # ----- debugging -----
    print(f"Zarr time stamps: {ds.time.values}")
    for v in ('elevation','slope','aspect'):
        da = ds[v]
        print(f"{v!r}: dims = {da.dims}, shape = {tuple(da.shape)}")
    for v in ('elevation','slope','aspect'):
        arr = ds[v].isel(time=0).values if 'time' in ds[v].dims else ds[v].values
        print(
            f"{v!r}: dtype={arr.dtype}, min={np.nanmin(arr):.3f}, "
            f"max={np.nanmax(arr):.3f}, NaNs={int(np.isnan(arr).sum())}"
        )
    list_vars(creds_path=args.aws_creds, 
              bucket=args.bucket, 
              folder=args.folder,
              geohash=args.geohash)
    logging.info(f"Elev (t0) min/max: {ds['elevation'].isel(time=0).min().values}, \
        {ds['elevation'].isel(time=0).max().values}")
    # ----- debugging -----

    inspect_tile_from_ds(
        ds,
        idx0=idx0,
        idx1=idx1,
        time0=time0,
        time1=time1,
        patch_loc=patch_loc,
        patch_size=args.patch_size,
        save_path=args.output,
        geohash=args.geohash,
        tiles_per_slice=tiles_per_slice
    )

if __name__=="__main__":
    main()

# Example:
# python inspect_dataset.py \
# 	--bucket rgc-zarr-store \
# 	--folder data \
# 	--geohashes 9v1z2 \
# 	--output outputs \
# 	--aws-creds secrets/aws_rgc-zarr-store.json \
# 	--dates first last

# yields
# inspections/rgb.png
# inspections/ndvi.png
# inspections/static.png
# inspections/summary_statistics.png
# inspections/report.pdf

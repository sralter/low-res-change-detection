# inspect_dataset.py

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
    if creds_path:
        return json.load(open(creds_path))
    return None

def list_vars(creds_path: str, 
              bucket: str, 
              folder: str,
              geohash: str):
    """
    Lists vars in dataset from S3
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
    s3_path = f"{bucket}/{folder}/{geohash}/{geohash}_zarr/"
    opts = {}
    if creds:
        opts = {'key': creds['access_key'], 'secret': creds['secret_access_key']}
    fs = s3fs.S3FileSystem(anon=not bool(creds), **opts)
    mapper = fs.get_mapper(s3_path)
    return xr.open_zarr(mapper, consolidated=consolidated)

def compute_band_summary(ds: xr.Dataset) -> pd.DataFrame:
    """
    Computes summary statistics (min/max/mean/std) over (time, y, x) for
    each data_var in ds.

    Returns a DataFrame indexed by band name
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

# def compute_time_summary(time_da: xr.DataArray) -> pd.Series:
#     """
#     Return min/max/mean/std/median/count of a DataArray of datetimes
#     as nicely-formatted strings:
#       - min/max/mean/median → YYYY-MM-DD
#       - std                → e.g. "12 days 00:00:00"
#       - count              → integer
#     """
#     # turn into a pandas DatetimeIndex
#     idx = pd.to_datetime(time_da.values)

#     # compute timestamp stats
#     tmin   = idx.min()
#     tmax   = idx.max()
#     tmean  = idx.mean()   # returns a Timestamp
#     tstd   = idx.std()    # returns a Timedelta
#     # for median we need a Series
#     tmedian = pd.Series(idx).median()
#     tcount  = idx.size

#     # format back to YYYY-MM-DD (and std as string)
#     return pd.Series({
#         "min":    tmin.strftime("%Y-%m-%d"),
#         "max":    tmax.strftime("%Y-%m-%d"),
#         "mean":   "N/A",#tmean.strftime("%Y-%m-%d"),
#         "std":    "N/A",#str(tstd),
#         "median": "N/A",#tmedian.strftime("%Y-%m-%d"),
#         "count":  tcount
#     })
def compute_time_summary(time_da: xr.DataArray) -> pd.Series:
    """
    Return min/max/count of a DataArray of datetimes as strings,
    formatting via _fmt_date to handle numpy, Python, or cftime objects.
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
    """Prepares PDF and out_dir paths"""
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

def detect_dims_bands(ds: xr.Dataset, show_ndvi: bool):
    """Docstring here"""
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

def compute_tile_indices(patch_size: int, ds, lat, lon, tdim, idx0: int):
    """Computes tile indices"""
    logging.info('Computing tile indices')
    H, W = ds.sizes[lat], ds.sizes[lon]
    def starts(n):
        s = list(range(0, n-patch_size+1, patch_size))
        if s and s[-1]+patch_size < n:
            s.append(n-patch_size)
        return s
    rs, cs = starts(H), starts(W)
    R, C = len(rs), len(cs)

    T = ds.sizes[tdim]
    time_pairs = [(t, t+1) for t in range(T-1)]
    total = len(time_pairs)*R*C
    if idx0 < 0:
        idx0 = 0
    elif idx0 >= total:
        idx0 = total - 1
    # if not (0 <= idx0 < total):
    #     raise IndexError(f"index {idx0} out of [0,{total})")

    pair_idx = idx0 // (R*C)
    tt0, tt1 = time_pairs[pair_idx]
    tile_idx = idx0 % (R*C)
    r0, c0   = rs[tile_idx//C], cs[tile_idx % C]
    r1, c1   = r0+patch_size, c0+patch_size
    return H, W, rs, cs, R, C, T, pair_idx, tt0, tt1, tile_idx, r0, c0, r1, c1, total, time_pairs

def make_starts(full: int, patch: int, stride: int) -> list[int]:
    starts = list(range(0, full - patch + 1, stride))
    if starts[-1] + patch < full:
        starts.append(full - patch)
    return starts

def resolve_time_indices(time0: str, time1: str, tdim, ds: xr.Dataset, tt0, tt1):
    if not tt0 <= ds[tdim].values.shape[0]:
        tt0 = ds[tdim].values.shape[0] - 1
        tt1 = tt0 + 1
    # if the user already passed integer indices, just return them
    if isinstance(time0, int) and isinstance(time1, int):
        return time0, time1
    # otherwise interpret time0/time1 as dates, find nearest in ds[tdim]
    if time0 is not None and time1 is not None:
        date0 = np.datetime64(time0)                    # e.g. "2019-01-01"
        date1 = np.datetime64(time1)
        times = ds[tdim].values                         # array of np.datetime64
        tt0 = int(np.argmin(np.abs(times - date0)))      # pick nearest index
        tt1 = int(np.argmin(np.abs(times - date1)))
        return tt0, tt1
    # fallback: use the tt0/tt1 provided by the caller
    return tt0, tt1

# def resolve_time_indices_(
#     time0: str | int | None,
#     time1: str | int | None,
#     tdim: str,
#     ds: xr.Dataset,
#     tt0: int,
#     tt1: int,
# ) -> tuple[int,int]:
#     """
#     Return a valid (t0, t1) pair, either from the ints the user gave
#     or (if they passed ISO strings) from the nearest dates in ds[tdim].
#     Always ensures:
#       0 <= t0 < T-1
#       t0 < t1 <= T-1
#     """
#     # total number of time steps
#     T = ds[tdim].size

#     # 1) clamp the fallback pair into [0,T-2]→[1,T-1]
#     tt0 = max(0, min(tt0, T-2))
#     tt1 = max(tt0+1, min(tt1, T-1))

#     # 2) if user literally passed integers to time0/time1, use those (clamped too)
#     if isinstance(time0, int) and isinstance(time1, int):
#         i0 = max(0, min(time0, T-2))
#         i1 = max(i0+1,     min(time1, T-1))
#         return i0, i1

#     # 3) if user passed ISO‐dates, snap to nearest indices
#     if isinstance(time0, str) and isinstance(time1, str):
#         arr = ds[tdim].values.astype("datetime64[D]")
#         d0  = np.datetime64(time0, "D")
#         d1  = np.datetime64(time1, "D")
#         i0  = int(np.argmin(np.abs(arr - d0)))
#         i1  = int(np.argmin(np.abs(arr - d1)))
#         # enforce a proper forward step
#         i0 = max(0, min(i0, T-2))
#         i1 = max(i0+1, min(i1, T-1))
#         return i0, i1

#     # 4) fallback to the clamped tt0,tt1
#     return tt0, tt1

def resolve_time_indices_(
    time0: str | int | None,
    time1: str | int | None,
    tdim: str,
    ds: xr.Dataset,
    tt0: int,
    tt1: int,
) -> tuple[int,int]:
    """
    Given optional dates (ISO-string or datetime.date) or None, plus fallback ints tt0/tt1,
    return a valid (t0, t1) within [0, T–1], enforcing t0 < t1.
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
    Slice out a (H,W) or (T,H,W) patch of `var` from `ds`:
      - spatial bounds [r0:r1, c0:c1] along lat_dim & lon_dim
      - optional time index t along time_dim
    """
    indexers: dict[str, slice | int] = {
        lat: slice(r0, r1),
        lon: slice(c0, c1),
    }
    if t is not None:
        indexers[time_dim] = t
    return ds[var].isel(indexers).values

# def rgb_before_after(lat, lon, r0, r1, c0, c1, tdim, ds: xr.Dataset, index: int,
#                      rgb_bands, t0, t1, total, save_path, pdf, out_dir, tt0, tt1,
#                      R, C):
#     """"""
#     logging.info('Setting up RGB before and after')

#     # def get_patch(var, t=None):
#     #     sel = {lat: slice(r0,r1), lon: slice(c0,c1)}
#     #     if t is not None: sel[tdim] = t
#     #     return ds[var].isel(sel).values

#     # compute tile‐grid row/col
#     tile_idx = index % (R*C)
#     row = tile_idx // C
#     col = tile_idx % C

#     # compute lat/lon bounds for the patch
#     lat_vals = ds[lat].values
#     lon_vals = ds[lon].values
#     lat_min, lat_max = lat_vals[r0],   lat_vals[r1-1]
#     lon_min, lon_max = lon_vals[c0],   lon_vals[c1-1]

#     p0 = np.stack([get_patch(b, t0) for b in rgb_bands], axis=0)
#     p1 = np.stack([get_patch(b, t1) for b in rgb_bands], axis=0)

#     def stretch(img):
#         lo, hi = np.percentile(img, (1,99))
#         return np.clip((img-lo)/(hi-lo+1e-6), 0,1)
#     def to_rgb(arr):
#         return stretch(np.moveaxis(arr,0,-1))

#     rgb0, rgb1 = to_rgb(p0), to_rgb(p1)
#     # fig, axes = plt.subplots(1,2,figsize=(12,6), constrained_layout=True)
#     # fig.suptitle(f"Patch {index+1} of {total}  •  times {tt0} → {tt1}", fontsize=16, y=1.02)
#     # for ax, img, t in zip(axes, (rgb0,rgb1),(tt0,tt1)):
#     ds_dates = ds[tdim].values
#     d0 = np.datetime_as_string(ds_dates[t0], unit='D')
#     d1 = np.datetime_as_string(ds_dates[t1], unit='D')
#     fig, axes = plt.subplots(1,2,figsize=(12,6), constrained_layout=True)
#     fig.suptitle(f"Patch {index+1} of {total}  •  dates {d0} → {d1}",
#                  fontsize=16, y=1.02)
#     for ax, img, t, d in zip(axes, (rgb0, rgb1), (t0, t1), (d0, d1)):
#         ax.imshow(img)
#         ax.get_xaxis().set_ticks([])
#         ax.get_yaxis().set_ticks([])
#         # timestamp label
#         # ts = np.datetime_as_string(ds[tdim].values[t], unit='s')
#         ts = d # label with just the date
 
#         # metadata box
#         meta = (
#             f"row={row}, col={col}\n"
#             f"lat: {lat_min:.4f} → {lat_max:.4f}\n"
#             f"lon: {lon_min:.4f} → {lon_max:.4f}"
#         )
#         ax.set_xlabel(meta, fontsize=11)
#         # ax.set_title(f"t={t}\n{ts}", fontsize=13)
#         ax.set_title(f"t={t}\n{ts}", fontsize=13)

#     if save_path:
#         out_png = out_dir / f"rgb_{index}.png"
#         fig.savefig(out_png, bbox_inches='tight')
#         pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
#     plt.close(fig)

def compute_human_readable_dates(ds: xr.Dataset, tdim, tt0, total, tt1):
    # compute human‐readable dates
    times = ds[tdim].values
    if tt0 <= total or tt1 <= total:
        d0, d1 = (np.datetime_as_string(times[t], unit="D") for t in (tt0, tt1))
    else:
        tt0 = total - 1
        tt1 = total
        d0, d1 = (np.datetime_as_string(times[t], unit="D") for t in (tt0, tt1))
    return d0, d1

# def time_indices_to_dates(
#     ds: xr.Dataset,
#     idx0, idx1,
# ) -> tuple[str | datetime.date, str | datetime.date]:
#     """Reverse of dates_to_time_indices"""
#     logging.info("Calculating dates from time indices...")
#     date0 = ds['time'].values[idx0]
#     date0 = np.datetime_as_string(date0, unit="D")
#     date1 = ds['time'].values[idx1]
#     date1 = np.datetime_as_string(date1, unit="D")
#     print("date0:", date0, '\n', "date1:", date1)
#     return date0, date1

def _fmt_date(t) -> str:
    """
    Cope with numpy.datetime64, Python datetime, and cftime objects.
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
    """Reverse of dates_to_time_indices"""
    logging.info("Calculating dates from time indices...")

    # pull out the raw time objects
    t0 = ds["time"].isel(time=idx0).item()
    t1 = ds["time"].isel(time=idx1).item()

    date0 = _fmt_date(t0)
    date1 = _fmt_date(t1)

    print("date0:", date0, "\ndate1:", date1)
    return date0, date1

# def rgb_before_after_(
#     ds: xr.Dataset,
#     rgb_bands: list[str],
#     r0: int, r1: int,
#     c0: int, c1: int,
#     tt0: int, tt1: int,
#     tdim: str,
#     lat: str, lon: str,
#     idx0: int, idx1: int, total: int,
#     out_dir: Path, pdf: PdfPages,
#     tag: str,
#     geohash: str
# ):
#     # compute human‐readable dates
#     # times = ds[tdim].values
#     # if tt0 <= total or tt1 <= total:
#     #     d0, d1 = (np.datetime_as_string(times[t], unit="D") for t in (tt0, tt1))
#     # else:
#     #     tt0 = total - 1
#     #     tt1 = total
#     #     d0, d1 = (np.datetime_as_string(times[t], unit="D") for t in (tt0, tt1))
#     # grab the raw time values at those two indices
#     tvals = ds[tdim].values
#     if tt0 > len(tvals) - 1 or tt1 > len(tvals) - 1:
#         tt0, tt1 = len(tvals) - 2, len(tvals) - 1
#     d0 = _fmt_date(tvals[tt0])
#     d1 = _fmt_date(tvals[tt1])

#     # stack patches
#     p0 = np.stack([
#         get_patch(ds, b, r0, r1, c0, c1, time_dim=tdim, t=tt0, lat=lat, lon=lon)
#         for b in rgb_bands
#     ], axis=0)
#     p1 = np.stack([
#         get_patch(ds, b, r0, r1, c0, c1, time_dim=tdim, t=tt1, lat=lat, lon=lon)
#         for b in rgb_bands
#     ], axis=0)

#     def stretch(img):
#         lo, hi = np.percentile(img, (1,99))
#         return np.clip((img - lo)/(hi - lo + 1e-6), 0,1)

#     rgb0, rgb1 = map(lambda arr: stretch(np.moveaxis(arr,0,-1)), (p0, p1))

#     date0, date1 = time_indices_to_dates(ds, idx0, idx1)

#     fig, axes = plt.subplots(1,2,figsize=(12,6), constrained_layout=True)
#     fig.suptitle(f"Geohash: {geohash}\nPatch {idx0+1} of {total} • {date0} to {date1}", y=1.02)

#     for ax, img, t, d in zip(axes, (rgb0, rgb1), (tt0, tt1), (date0, date1)):
#         ax.imshow(img)
#         ax.axis("off")
#         ax.set_title(d, fontsize=13)

#     if out_dir:
#         fname = out_dir / f"rgb_{tag}.png"
#         fig.savefig(fname, bbox_inches="tight")
#         pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
#     plt.close(fig)
# def rgb_before_after_(
#     ds: xr.Dataset,
#     rgb_bands: list[str],
#     r0: int, r1: int,
#     c0: int, c1: int,
#     tt0: int, tt1: int,
#     tdim: str,
#     lat: str, lon: str,
#     idx0: int, idx1: int, total: int,
#     out_dir: Path, pdf: PdfPages,
#     tag: str,
#     geohash: str
# ):
#     # --- human‐readable dates as before ---
#     tvals = ds[tdim].values
#     if tt0 > len(tvals)-1 or tt1 > len(tvals)-1:
#         tt0, tt1 = len(tvals)-2, len(tvals)-1
#     d0 = _fmt_date(tvals[tt0])
#     d1 = _fmt_date(tvals[tt1])
#     date0, date1 = time_indices_to_dates(ds, idx0, idx1)

#     # --- helper to stretch for display ---
#     def stretch(img):
#         lo, hi = np.percentile(img, (1,99))
#         return np.clip((img - lo)/(hi - lo + 1e-6), 0,1)

#     # --- build full‐frame RGB for each time ---
#     full0 = np.stack([
#         ds[b].isel({tdim:tt0}).values
#         for b in rgb_bands
#     ], axis=0)
#     full1 = np.stack([
#         ds[b].isel({tdim:tt1}).values
#         for b in rgb_bands
#     ], axis=0)
#     rgb_full0 = stretch(np.moveaxis(full0, 0, -1))
#     rgb_full1 = stretch(np.moveaxis(full1, 0, -1))

#     # --- build patch RGB as before ---
#     p0 = np.stack([
#         get_patch(ds, b, r0, r1, c0, c1,
#                   time_dim=tdim, t=tt0, lat=lat, lon=lon)
#         for b in rgb_bands
#     ], axis=0)
#     p1 = np.stack([
#         get_patch(ds, b, r0, r1, c0, c1,
#                   time_dim=tdim, t=tt1, lat=lat, lon=lon)
#         for b in rgb_bands
#     ], axis=0)
#     rgb0 = stretch(np.moveaxis(p0, 0, -1))
#     rgb1 = stretch(np.moveaxis(p1, 0, -1))

#     # --- set up 2×2 figure: full before/after, patch before/after ---
#     fig, axes = plt.subplots(2, 2, figsize=(12,12), constrained_layout=True)
#     fig.suptitle(
#         f"Geohash: {geohash}\nPatch {idx0+1} of {total} • {date0} → {date1}",
#         y=1.02
#     )

#     # top‐row: full images with red rectangle
#     for ax, img_full, d in zip(axes[0], (rgb_full0, rgb_full1), (d0, d1)):
#         ax.imshow(img_full)
#         ax.axis("off")
#         ax.set_title(f"Full frame: {d}", fontsize=13)
#         # draw a red box at (c0, r0) of width=c1-c0, height=r1-r0
#         rect = Rectangle(
#             (c0, r0),
#             c1 - c0,
#             r1 - r0,
#             edgecolor="red",
#             facecolor="none",
#             linewidth=2
#         )
#         ax.add_patch(rect)

#     # bottom‐row: the original patches
#     for ax, img_patch, d in zip(axes[1], (rgb0, rgb1), (d0, d1)):
#         ax.imshow(img_patch)
#         ax.axis("off")
#         ax.set_title(f"Patch", fontsize=13)

#     # --- save & close ---
#     if out_dir:
#         fname = out_dir / f"rgb_{tag}.png"
#         fig.savefig(fname, bbox_inches="tight")
#         pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
#     plt.close(fig)
def rgb_before_after_(
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

# def ndvi_comparison(
#     ds: xr.Dataset,
#     ndvi_band: str,
#     r0: int, r1: int,
#     c0: int, c1: int,
#     tt0: int, tt1: int,
#     tdim: str,
#     lat: str, lon: str,
#     date0, date1,
#     idx0, idx1,
#     out_dir: Path, pdf: PdfPages,
#     tag: str
# ):
#     nd0 = get_patch(ds, ndvi_band, r0, r1, c0, c1,
#                     time_dim=tdim, t=tt0, lat=lat, lon=lon)
#     nd1 = get_patch(ds, ndvi_band, r0, r1, c0, c1,
#                     time_dim=tdim, t=tt1, lat=lat, lon=lon)
#     diff = nd1 - nd0

#     fig, axes = plt.subplots(1,3,figsize=(15,5), constrained_layout=True)
#     titles = (
#         f"NDVI t₀\nIndex: {idx0}, Date: {date0}",
#         f"NDVI t₁\nIndex: {idx1}, Date: {date1}",
#         "Δ NDVI\n"
#     )
#     for ax, arr, title in zip(axes, (nd0, nd1, diff), titles):
#         im = ax.imshow(arr, vmin=-1, vmax=1)
#         ax.set_title(title)
#         ax.axis("off")
#         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
#     fig.suptitle("NDVI Comparison")

#     if out_dir:
#         fname = out_dir / f"ndvi_{tag}.png"
#         fig.savefig(fname, bbox_inches="tight")
#         pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
#     plt.close(fig)
def ndvi_before_after_(
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

# def static_band_maps(
#     ds: xr.Dataset,
#     static_bands: list[str],
#     r0: int, r1: int,
#     c0: int, c1: int,
#     tt0: int,
#     time_dim: str,
#     lat: str,
#     lon: str,
#     idx0: int, idx1: int,
#     total: int,
#     out_dir: Path,
#     pdf: PdfPages,
#     tag: str
# ):
#     """
#     1) For each static band (elevation, slope, aspect):
#        - pull the patch at time t0 if it's time-varying, otherwise across space.
#        - log its global min/max and patch min/max.
#        - percentile-stretch it and plot.
#        - save both PNG and add to PDF.
#     """
#     logging.info("Setting up static band maps")

#     # 1) log global min/max for each static band
#     for var in static_bands:
#         gmin = float(ds[var].min().compute().values)
#         gmax = float(ds[var].max().compute().values)
#         logging.info(f"Global {var} min/max: {gmin:.3f} → {gmax:.3f}")

#     # 2) now loop each band and make a plot of the local patch
#     static_bands_prefs = [ # static (elev, slo, asp), cmap, title
#         (static_bands[0], 'terrain', 'Elevation (m)'),
#         (static_bands[1], 'viridis', 'Slope (degrees)'),
#         (static_bands[2], 'viridis', 'Aspect (degrees)')
#     ]
#     for var, cmap, title in static_bands_prefs:
#         # pull patch; only slice time if the band actually has that dim
#         if time_dim in ds[var].dims:
#             raw = get_patch(
#                 ds, var,
#                 r0, r1, c0, c1,
#                 time_dim=time_dim, t=tt0,
#                 lat=lat, lon=lon
#             )
#         else:
#             raw = get_patch(
#                 ds, var,
#                 r0, r1, c0, c1,
#                 time_dim=None, t=None,
#                 lat=lat, lon=lon
#             )

#         # diagnostic stats on the patch
#         mn, mx = float(np.nanmin(raw)), float(np.nanmax(raw))
#         logging.info(f"[DEBUG] {var} patch min/max: {mn:.3f} → {mx:.3f}")

#         # # percentile‐stretch to [0,1]
#         # lo, hi = np.nanpercentile(raw, (1, 99))
#         # img = np.clip((raw - lo) / (hi - lo + 1e-6), 0, 1)
#         img = raw

#         # plot
#         fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
#         im = ax.imshow(img, cmap=cmap)
#         ax.set_title(title, fontsize=12)
#         ax.axis("off")
#         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

#         # save & add to PDF
#         if out_dir:
#             out_png = out_dir / f"{var}_{tag}.png"
#             fig.savefig(out_png, bbox_inches="tight")
#             pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)

#         plt.close(fig)
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
    Show a 2x3 grid:
    full elevation  | full slope  | full aspect
    patch elevation | patch slope | patch aspect
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
    Compute summary stats for time + every data_var, render as a table,
    add it as the final page of the PDF (but does NOT close the PDF).
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
    1) RGB before/after
    2) NDVI before/after + diff
    3) Static-band maps (elevation, slope, aspect)
    Also, if save_path is given, writes PNGs *and* a PDF report.
    """
    # --- detect dims & bands ---
    Dims, tdim, lat, lon, rgb_bands, ndvi_band, static_bands = detect_dims_bands(
        ds=ds, show_ndvi=show_ndvi)

    # # --- determine time indices (explicit dates wins, otherwise index‐based) ---
    # if time0 is not None and time1 is not None:
    #     # use the provided time indices directly
    #     tt0, tt1 = time0, time1
    #     # now compute just the tile indices and grid dims for that index
    #     H, W = ds.sizes[lat], ds.sizes[lon]
    #     rs = list(range(0, H-patch_size+1, patch_size))
    #     cs = list(range(0, W-patch_size+1, patch_size))
    #     R, C = len(rs), len(cs)
    #     total = R*C  # no time‐pairs, so just spatial tiles
    #     pair_idx = None
    #     tile_idx = idx0 % (R*C)
    #     r0, c0 = rs[tile_idx//C], cs[tile_idx % C]
    #     r1, c1 = r0 + patch_size, c0 + patch_size
    # else:
    #     # fallback to your original “next‐time” + patch‐grid logic
    #     (H, W, rs, cs, R, C,
    #      T, pair_idx, tt0, tt1,
    #      tile_idx, r0, c0, r1, c1,
    #      total, time_pairs) = compute_tile_indices(
    #         patch_size=patch_size, ds=ds, lat=lat, lon=lon,
    #         tdim=tdim, idx0=idx0
    #     )
    # --- time indices already resolved in main()
    tt0, tt1 = idx0, idx1

    # --- compute patch row/col from user patch_loc ---
    H, W = ds.sizes[lat], ds.sizes[lon]
    rows = make_starts(H, patch_size, patch_size)
    cols = make_starts(W, patch_size, patch_size)
    # rs = list(range(0, H - patch_size + 1, patch_size))
    # cs = list(range(0, W - patch_size + 1, patch_size))
    # R, C = len(rs), len(cs)
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
    # tt0, tt1 = resolve_time_indices(time0=time0, time1=time1, tdim=tdim, ds=ds,
    #                               tt0=idx0, tt1=idx1)
    tt0, tt1 = resolve_time_indices_(time0, time1, tdim, ds, tt0, tt1)
    print(f"tt0, tt1: {tt0}, {type(tt0)}, {tt1}, {type(tt1)}")

    # --- 1) RGB before/after ---
    # rgb_before_after(lat=lat, lon=lon, r0=r0, r1=r1, c0=c0, c1=c1, tdim=tdim,
    #                  ds=ds, index=index, rgb_bands=rgb_bands, t0=t0, t1=t1,
    #                  total=total, save_path=save_path, pdf=pdf, out_dir=out_dir,
    #                  tt0=tt0, tt1=tt1, R=R, C=C)

    rgb_before_after_(ds=ds, rgb_bands=rgb_bands, r0=r0, r1=r1, c0=c0, c1=c1, 
                      tt0=tt0, tt1=tt1, tdim=tdim, lat_dim=lat, lon_dim=lon,
                      idx0=idx0, idx1=idx1, total=total, patch_loc=patch_loc,
                      out_dir=out_dir, pdf=pdf, tag=tag, geohash=geohash, 
                      tiles_per_slice=tiles_per_slice)

    # def get_patch(var, t=None):
    #     sel = {lat: slice(r0,r1), lon: slice(c0,c1)}
    #     if t is not None: sel[tdim] = t
    #     return ds[var].isel(sel).values

    # # compute tile‐grid row/col
    # tile_idx = index % (R*C)
    # row = tile_idx // C
    # col = tile_idx % C

    # # compute lat/lon bounds for the patch
    # lat_vals = ds[lat].values
    # lon_vals = ds[lon].values
    # lat_min, lat_max = lat_vals[r0],   lat_vals[r1-1]
    # lon_min, lon_max = lon_vals[c0],   lon_vals[c1-1]

    # p0 = np.stack([get_patch(b, tt0) for b in rgb_bands], axis=0)
    # p1 = np.stack([get_patch(b, tt1) for b in rgb_bands], axis=0)

    # def stretch(img):
    #     lo, hi = np.percentile(img, (1,99))
    #     return np.clip((img-lo)/(hi-lo+1e-6), 0,1)
    # def to_rgb(arr):
    #     return stretch(np.moveaxis(arr,0,-1))

    # rgb0, rgb1 = to_rgb(p0), to_rgb(p1)
    # fig, axes = plt.subplots(1,2,figsize=(12,6), constrained_layout=True)
    # fig.suptitle(f"Patch {index+1} of {total}  •  times {tt0} → {tt1}", fontsize=16, y=1.02)
    # for ax, img, t in zip(axes, (rgb0,rgb1),(tt0,tt1)):
    #     ax.imshow(img)
    #     ax.get_xaxis().set_ticks([])
    #     ax.get_yaxis().set_ticks([])
    #     # timestamp label
    #     ts = np.datetime_as_string(ds[tdim].values[t], unit='s')

    #     # metadata box
    #     meta = (
    #         f"row={row}, col={col}\n"
    #         f"lat: {lat_min:.4f} → {lat_max:.4f}\n"
    #         f"lon: {lon_min:.4f} → {lon_max:.4f}"
    #     )
    #     ax.set_xlabel(meta, fontsize=11)
    #     ax.set_title(f"t={t}\n{ts}", fontsize=13)

    # if save_path:
    #     out_png = out_dir / f"rgb_{index}.png"
    #     fig.savefig(out_png, bbox_inches='tight')
    #     pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
    # plt.close(fig)

    # --- 2) NDVI panels ---
    # ndvi_comparison(ds, ndvi_band, r0, r1, c0, c1, tt0, tt1, tdim, lat, lon, 
    #                 date0, date1, idx0, idx1, out_dir, pdf, tag)
    ndvi_before_after_(ds=ds, ndvi_band=ndvi_band, r0=r0, r1=r1, c0=c0, c1=c1, 
                       tt0=tt0, tt1=tt1, tdim=tdim, lat_dim=lat, lon_dim=lon,
                       idx0=idx0, idx1=idx1, total=total, patch_loc=patch_loc,
                       out_dir=out_dir, pdf=pdf, tag=tag, geohash=geohash,
                       tiles_per_slice=tiles_per_slice)

    # logging.info('Setting up NDVI panels')
    # if show_ndvi:
    #     nd0 = get_patch(ndvi_band, tt0)
    #     nd1 = get_patch(ndvi_band, tt1)
    #     diff = nd1 - nd0
    #     fig, axes = plt.subplots(1,3,figsize=(15,5), constrained_layout=True)
    #     for ax, arr, title in zip(
    #         axes, (nd0,nd1,diff), ('NDVI t0','NDVI t1','Δ NDVI')
    #     ):
    #         im = ax.imshow(arr, vmin=-1, vmax=1)
    #         ax.set_title(title); ax.axis('off')
    #         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    #     if save_path:
    #         png0 = out_dir / f"ndvi_{index}.png"
    #         # png1 = out_dir / f"ndvi_diff_{index}.png"
    #         fig.savefig(png0, bbox_inches='tight')
    #         # fig.savefig(png1, bbox_inches='tight')
    #         pdf.savefig(fig)
    #     plt.close(fig)

    # --- 3) Static-band maps ---
    static_band_maps(ds=ds, static_bands=static_bands, r0=r0, r1=r1, c0=c0, c1=c1, 
                     tt0=tt0, lat_dim=lat, lon_dim=lon, idx0=idx0, 
                     total=total, out_dir=out_dir, pdf=pdf, tag=tag, geohash=geohash, 
                     tiles_per_slice=tiles_per_slice, patch_loc=patch_loc, 
                     date0=date0, date1=date1)

    # logging.info('Setting up static band maps')
    # # compute global min/max into NumPy scalars
    # elev_min = float(ds['elevation'].min().compute().values)
    # elev_max = float(ds['elevation'].max().compute().values)
    # logging.info(f"Global elev min/max: {elev_min:.3f} → {elev_max:.3f}")
    # raw = get_patch('elevation', tt0)
    # n_total = raw.size
    # n_nan   = np.isnan(raw).sum()
    # logging.info(f"Elevation patch: {n_total - n_nan} valid pixels out of {n_total}")
    # for var in static_bands:
    #     # 1) grab raw 2D
    #     if tdim in ds[var].dims:
    #         raw = get_patch(var, tt0)
    #     else:
    #         raw = get_patch(var)

    #     # 2) diagnostic: print its range
    #     mn, mx = np.nanmin(raw), np.nanmax(raw)
    #     logging.info(f"[DEBUG] {var} patch range: {mn:.3f} → {mx:.3f}")

    #     # 3) percentile-stretch to [0,1] so we actually see variation
    #     lo, hi = np.nanpercentile(raw, (1, 99))
    #     img = np.clip((raw - lo) / (hi - lo + 1e-6), 0, 1)

    #     # 4) plot
    #     fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    #     im = ax.imshow(img)
    #     ax.set_title(var, fontsize=12)
    #     ax.axis("off")
    #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    #    # 5) save & add to PDF
    #     if save_path:
    #         out_png = out_dir / f"{var}_{index}.png"
    #         fig.savefig(out_png, bbox_inches="tight")
    #         pdf.savefig(fig)
    #     plt.close(fig)

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
    Given ds[time_dim] (an array of np.datetime64's) and two dates
    (either ISO strings 'YYYY-MM-DD' or date/datetime), return the
    integer indices of the points in ds.time nearest to each.
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
    # group.add_argument("--dates", "-d", nargs=2, type=datetime.date.fromisoformat,
    #                    metavar=("DATE0", "DATE1"),
    #                    help="Inspect two arbitrary dates (YYYY-MM-DD YYYY-MM-DD)")
    return p.parse_args()

def main():
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
    
    # parse dates
    # if args.dates is not None:
    #     # user called --dates yyyy-mm-dd yyyy-mm-dd
    #     # two dates → find nearest integer indices
    #     time0, time1 = args.dates
    #     idx0, idx1 = dates_to_time_indices(ds, args.dates)
    # else:
    #     # user called --index N
    #     # single index → index and index+1 as our window
    #     idx0, idx1 = args.index, args.index + 1
    #     time0 = time1 = None
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

# ./inspect_zarr.py \
#   -b my-bucket \
#   -f data \
#   -g 9vgm6 \
#   -o output \
#   -a secrets/aws_rgc-zarr-store.json \
#   -d 2018-12-01 2019-12-01

# yields
# inspections/run42_tile17_rgb_17.png
# inspections/run42_tile17_ndvi_17.png
# inspections/run42_tile17_ndvi_diff_17.png
# inspections/run42_tile17_elevation_17.png
# inspections/run42_tile17_slope_17.png
# inspections/run42_tile17_aspect_17.png
# inspections/run42_tile17_17_report.pdf

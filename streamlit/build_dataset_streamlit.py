# build_dataset_streamlit.py

import argparse
import json
from datetime import datetime, timezone, timedelta
import geohash2
import ee, geemap
from pathlib import Path
import tempfile
import csv
from typing import Callable, Optional
import xarray as xr
import rasterio as rio
import rioxarray as rxr
import pandas as pd
import numpy as np
import zarr.convenience
# monkey patch to avoid np.product/np.prod error with Zarr
np.product = np.prod
import s3fs
import fsspec
import sys
import zarr
import math
import gc
from affine import Affine
import logging
from osgeo import gdal, osr
import pymaap
pymaap.init_general_logger()
# import warnings
# import rasterio as rio

# warnings.filterwarnings(module="rasterio", action="ignore", message=".*TIFFReadDirectory:Sum of Photometric.*")
# warnings.filterwarnings(action="ignore", message=".*explicit representation of timezones .*")

# Reduce noise from dependencies
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("aiobotocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("s3fs").setLevel(logging.WARNING)
logging.getLogger("rasterio").setLevel(logging.WARNING)

def authenticate_ee(account_key: str = None, account_email: str = None):
    """
    Authenticates Google Earth Engine: if both account_key and account_email
    are provided, uses a service account; otherwise falls back to default/interactive.
    """
    if account_key and account_email:
        key_path = Path(account_key)
        if not key_path.exists():
            logging.error("GEE key file not found at %s", key_path)
            raise FileNotFoundError(f"Key file not found: {key_path}")
        creds = ee.ServiceAccountCredentials(account_email, str(key_path))
        ee.Initialize(credentials=creds)
        print("GEE successfully initialized using service account credentials.")
    else:
        try:
            ee.Initialize()
            print("GEE initialized with default credentials.")
        except Exception:
            ee.Authenticate()
            ee.Initialize()
            print("GEE authenticated interactively.")

def create_bbox(geohash: str) -> dict:
    """Creates bounding box from geohash"""
    # logging.info(f"Create bounding box from geohash {geohash}")
    lat, lon, lat_err, lon_err = geohash2.decode_exactly(geohash)
    bbox = {
        'xmin': lon - lon_err,
        'ymin': lat - lat_err,
        'xmax': lon + lon_err,
        'ymax': lat + lat_err
    }
    return bbox

def create_ee_bbox(bbox) -> ee.Geometry.BBox:
    """Creates Earth Engine-friendly bounding box from basic bounding box"""
    logging.info(f"Creating Earth Engine-friendly bounding box")
    region = ee.Geometry.BBox(
        bbox['xmin'], bbox['ymin'],
        bbox['xmax'], bbox['ymax']
    )
    return region

def get_ee_region(geohash: str) -> ee.Geometry.BBox:
    """Combines create_bbox and create_ee_bbox"""
    bbox = create_bbox(geohash)
    region = create_ee_bbox(bbox)
    return region

def get_ee_collection(region: ee.Geometry.BBox, 
                      cloud_perc: int) -> ee.ImageCollection:
    """
    Outputs Earth Engine Image Collection based on given region 
    and cloud percentage threshold filter (defaults to 5).
    """
    logging.info(f"Getting Earth Engine Image Collection with cloudy percentage below {cloud_perc}")
    coll_all = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))
    return coll_all

def build_s3_path(geohash: str, bucket: str, folder: str) -> str:
    """Builds S3 path for S3FS"""
    bucket = bucket.strip("/")
    folder = folder.strip("/")
    base_uri = f"s3://{bucket}/{folder}"
    geohash_folder = f"{base_uri}/{geohash}"
    zarr_path = f"{geohash_folder}/{geohash}_zarr/"
    climatology_path = f"{geohash_folder}/{geohash}_climatology"
    return base_uri, zarr_path, climatology_path

def make_aws_creds_dict(aws_secrets_file: str) -> dict:
    """Makes AWS credentials dictionary"""
    try:
        if aws_secrets_file:
            creds = json.load(open(aws_secrets_file))
            storage_options = {
                'key': creds['access_key'],
                'secret': creds['secret_access_key']
            }
    except Exception as e:
        logging.error(f"Error: {e} - Please provide a valid AWS Secrets JSON file")

    return storage_options

def check_existence(geohash: str, bucket: str, folder: str, storage_options: dict) -> bool:
    """Checks the existence of a Zarr at the specified S3 location."""
    # build S3 path
    _, zarr_path, _ = build_s3_path(geohash=geohash,
                                    bucket=bucket,
                                    folder=folder)
    # use s3fs to check existence
    fs = s3fs.S3FileSystem(**storage_options)
    exists = fs.exists(path=zarr_path)
    return exists, zarr_path

def get_zarr_date_range(zarr_path: str, 
                      storage_options: dict,
                      geohash: str) -> list[str]:
    """Gets date range of Zarr dataset from S3"""
    # open dataset
    # geohash = zarr_path.split('/')[2]
    fs = s3fs.S3FileSystem(**storage_options)
    mapper = fs.get_mapper(zarr_path)
    ds = xr.open_zarr(mapper, consolidated=False, storage_options=storage_options)
    
    # inspect end dates
    time = ds['time']
    existing_start_time = time.min().values.astype('datetime64[D]')
    existing_end_time = time.max().values.astype('datetime64[D]')

    logging.info(f"Existing date range for {geohash}: {existing_start_time} to {existing_end_time}")

    return existing_start_time, existing_end_time

def update_metadata_csv(metadata_csv: Path,
                        new_rows: list[dict],
                        unique_subset: list[str] = None):
    """
    - metadata_csv: path to your existing CSV
    - new_rows:    list of dicts, each matching your CSV columns
    - unique_subset: columns to identify duplicates (defaults to all columns)
    """
    # 1. Read existing (or start empty if not exist)
    if metadata_csv.exists():
        meta_df = pd.read_csv(metadata_csv, parse_dates=["datetime"])
    else:
        meta_df = pd.DataFrame(columns=new_rows[0].keys())

    # 2. Build the new‐rows DataFrame
    new_df = pd.DataFrame(new_rows)
    # ensure datetime is a datetime dtype
    new_df["datetime"] = pd.to_datetime(new_df["datetime"])

    # 3. Concat, drop exact duplicates, sort by datetime
    combined = pd.concat([meta_df, new_df], ignore_index=True)
    combined.drop_duplicates(subset=unique_subset or combined.columns.tolist(),
                             keep="first", inplace=True)
    combined.sort_values(by="datetime", inplace=True)

    # 4. Write back (atomic save via temp file is optional)
    combined.to_csv(metadata_csv, index=False)

def download_missing_for_geohash(
        gh: str, 
        region, 
        collection,
        window_start, 
        window_end, 
        out_dir, 
        scale: int,
        cloud_perc: int):
    """
    Downloads all Sentinel-2 images between window_start and window_end for one geohash.
    """
    logging.info(f"[{gh}] Downloading images {window_start} → {window_end}")
    coll = collection.filterDate(
        window_start.strftime("%Y-%m-%d"),
        (window_end + timedelta(days=1)).strftime("%Y-%m-%d")
    )
    
    features = coll.getInfo()["features"]
    for feat in features:
        props = feat["properties"]
        ts   = props["system:time_start"]
        dt   = datetime.fromtimestamp(ts/1000).strftime("%Y%m%d")
        fname = out_dir / f"{gh}_{dt}.tif"
        
        comp = ee.Image(feat["id"]) \
                 .select(["B4","B3","B2"]).rename(["R","G","B"]) \
                 .addBands(ee.Image("USGS/SRTMGL1_003")
                         .clip(region)
                         .normalizedDifference(['B8','B4']).rename("NDVI")) \
                 .addBands(ee.Terrain.slope(
                         ee.Image("USGS/SRTMGL1_003")).rename("slope")) \
                 .addBands(ee.Terrain.aspect(
                         ee.Image("USGS/SRTMGL1_003")).rename("aspect"))
        
        logging.info(f"  Exporting {fname.name}")
        geemap.ee_export_image(
            comp,
            filename=str(fname),
            scale=scale,
            region=region
        )

def compute_slope_aspect(
    elev: xr.DataArray,
    transform: Affine,
    nodata: float | None = None
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    From a 2D elevation DataArray, compute slope and aspect in degrees.
    """
    # pixel size
    xres = transform.a
    yres = -transform.e

    arr = elev.values.astype(float)
    if nodata is not None:
        arr[arr == nodata] = np.nan

    # ∂Z/∂y, ∂Z/∂x
    dz_dy, dz_dx = np.gradient(arr, yres, xres)

    # slope (deg)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    # aspect (deg clockwise from north)
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (aspect + 360) % 360

    coords = elev.coords
    return (
        xr.DataArray(slope, dims=elev.dims, coords=coords, name="slope"),
        xr.DataArray(aspect, dims=elev.dims, coords=coords, name="aspect"),
    )

def compute_slope_aspect_gdal(
    elev_da: xr.DataArray,
    transform: Affine,
    crs_epsg: int = 4326
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute slope & aspect via GDAL DEMProcessing on an in-memory raster.

    Parameters
    ----------
    elev_da : xr.DataArray
        2D elevation with dims (y, x).
    transform : Affine
        GeoTransform for that array.
    crs_epsg : int
        EPSG code for the CRS (default 4326).

    Returns
    -------
    slope_da, aspect_da : tuple of xr.DataArray
    """
    # 1) Create an in-memory GDAL dataset
    height, width = elev_da.shape
    mem_drv = gdal.GetDriverByName('MEM')
    ds = mem_drv.Create('', width, height, 1, gdal.GDT_Float32)

    # set geotransform & CRS
    ds.SetGeoTransform((transform.c, transform.a, transform.b,
                        transform.f, transform.d, transform.e))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs_epsg)
    ds.SetProjection(srs.ExportToWkt())

    # write elevation band
    band = ds.GetRasterBand(1)
    arr = elev_da.values.astype(float)
    band.WriteArray(arr)

    # explicit cast to Python float (and guard against None)
    ndv = elev_da.rio.nodata
    if ndv is not None:
        try:
            band.SetNoDataValue(float(ndv))
        except (TypeError, ValueError):
            # if it still fails, just warn and move on
            logging.warning(f"Could not set GDAL nodata to {ndv!r}")

    # 2) DEMProcessing to slope & aspect (in-memory)
    slope_ds  = gdal.DEMProcessing('', ds, 'slope',  format='MEM', computeEdges=True)
    aspect_ds = gdal.DEMProcessing('', ds, 'aspect', format='MEM', computeEdges=True)

    # 3) read back into numpy
    slope_arr  = slope_ds.GetRasterBand(1).ReadAsArray()
    aspect_arr = aspect_ds.GetRasterBand(1).ReadAsArray()

    # 4) wrap into DataArrays with the same coords/dims as elev_da
    coords = elev_da.coords
    slope_da  = xr.DataArray(slope_arr,  dims=elev_da.dims, coords=coords, name='slope')
    aspect_da = xr.DataArray(aspect_arr, dims=elev_da.dims, coords=coords, name='aspect')

    return slope_da, aspect_da

def compute_climatology(ds: xr.Dataset, save_dir: str, storage_options: dict = None):
    """
    Calculates seasonal and monthly NDVI layers and writes them to 
    save_dir (local or s3://).
    """
    # 1. Early exit if already done
    filenames = [
        "seasonal_ndvi.nc",
        "monthly_ndvi_mean.nc",
        "monthly_ndvi_std.nc",
    ]
    is_s3 = save_dir.startswith("s3://")
    if is_s3:
        fs = fsspec.filesystem("s3", **storage_options)
        prefix = save_dir.rstrip("/")
        if all(fs.exists(f"{prefix}/{fn}") for fn in filenames):
            print(f"Climatology already exists on S3 at {prefix}; skipping.")
            return
    else:
        os.makedirs(save_dir, exist_ok=True)
        if all((Path(save_dir)/fn).exists() for fn in filenames):
            print(f"Climatology already exists at {save_dir}; skipping.")
            return

    # 2. Make sure time is datetime64 and drop Feb-29
    ds = ds.assign_coords(time=("time", pd.to_datetime(ds["time"].values)))
    leap = (ds["time"].dt.month==2)&(ds["time"].dt.day==29)
    if leap.any():
        ds = ds.sel(time=~leap)

    # 3. Build grouping coords
    ds = ds.assign_coords(
        doy   = ds["time"].dt.dayofyear,
        month = ds["time"].dt.month
    )

    # Rename NDVI → ndvi (if needed)
    if "ndvi" not in ds and "NDVI" in ds:
        ds = ds.rename({"NDVI": "ndvi"})
    ndvi = ds["ndvi"]

    # 4. Compute
    outputs = {
        "seasonal_ndvi.nc":     ndvi.groupby("doy").mean("time"),
        "monthly_ndvi_mean.nc": ndvi.groupby("month").mean("time"),
        "monthly_ndvi_std.nc":  ndvi.groupby("month").std("time"),
    }

    # 5. Write local or S3
    if is_s3:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fs_s3 = fsspec.filesystem("s3", **storage_options)
            prefix = save_dir.rstrip("/")
            for fn, arr in outputs.items():
                tmp = td/fn
                arr.to_netcdf(tmp)
                fs_s3.put(str(tmp), f"{prefix}/{fn}")
    else:
        for fn, arr in outputs.items():
            arr.to_netcdf(Path(save_dir)/fn)

    print(f"Climatology files saved to {save_dir}")

def enrich_with_slope_aspect(
    in_tif: Path,
    transform: Affine,
    out_tif: Path,
) -> xr.DataArray:
    """
    Load a 5-band GeoTIFF (R,G,B,NDVI,elev), compute slope/aspect from band 5,
    add a mask channel, then write out an 8-band GeoTIFF and return it.
    """
    da5 = rxr.open_rasterio(str(in_tif), masked=False)
    if "band" in da5.dims and da5.sizes["band"] == 1:
        da5 = da5.squeeze("band", drop=True)

    # 1) split out elevation
    elev_da = da5.isel(band=4)

    # 2) compute slope & aspect (GDAL or numpy version)
    slope_da, aspect_da = compute_slope_aspect_gdal(elev_da, transform)
    # or: slope_da, aspect_da = compute_slope_aspect(elev_da, transform)

    # 3) mask: 1.0 where aspect is valid, 0.0 where it was nodata (or -9999)
    #    first replace any -9999 with NaN so that isnan() picks them up
    aspect_da = aspect_da.where(aspect_da != -9999, np.nan)
    mask_arr  = (~np.isnan(aspect_da.values)).astype(np.uint8)
    mask_da   = xr.DataArray(
        mask_arr,
        dims=aspect_da.dims,
        coords=aspect_da.coords,
        name="aspect_mask"
    )

    # 4) stitch into an 8-band array
    rgbndvi = da5.isel(band=[0,1,2,3])
    da8 = xr.concat(
        [rgbndvi, elev_da, slope_da, aspect_da, mask_da],
        dim="band"
    ).assign_coords(band=[
        "R", "G", "B", "NDVI",
        "elevation", "slope", "aspect", "aspect_mask"
    ])

    # 5) write out and return
    da8.rio.to_raster(str(out_tif))
    return da8

def log_raster_stats(
    da: xr.DataArray,
    label: str,
):
    arr = da.values
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    n_total = arr.size
    n_nan = int(np.isnan(arr).sum())
    logging.info(
        f"[{label}] shape={da.dims}:{da.shape}, "
        f"min={mn}, max={mx}, valid={n_total-n_nan}/{n_total}"
    )

def download_all_imagery_with_metadata(
    collection: ee.ImageCollection,
    out_dir: Path,
    bbox,
    scale: int,
    region: ee.Geometry,
    metadata_csv: Path,
    geohash: str,
    end_date: datetime.date = None,
    skip_static: bool = False
):
    """
    Downloads every image in `collection` (from its first date through now),
    builds a 7-band composite (R, G, B, NDVI, elevation, slope, aspect),
    writes each to GeoTIFF under out_dir, and logs one row per image (plus
    the three static layers) to metadata_csv.
    """
    # region bbox in projected units (e.g. degrees if EPSG:4326)
    xmin, ymin, xmax, ymax = (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])
    # scale in meters or degrees

    # calculate width/height in pixels
    width  = int((xmax - xmin) / scale)
    height = int((ymax - ymin) / scale)

    # build an Affine transform: (a, b, c, d, e, f) maps pixel to geocoords
    transform = Affine(scale, 0, xmin, 0, -scale, ymax)
    crs = "EPSG:4326"

    # guard empty EE collections
    n_images = collection.size().getInfo()
    if n_images == 0:
        logging.info(f"[{geohash}] no images in requested window → skipping download")
        return

    # 1. Make dirs & open CSV (append if it already exists)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "filename", "datetime", "system_index",
        "DATATAKE_IDENTIFIER", "CLOUDY_PIXEL_PERCENTAGE",
        "static_included"
    ]
    # if the file already exists, append; otherwise write (and emit header)
    mode = "a" if metadata_csv.exists() else "w"
    with open(metadata_csv, mode, newline="") as mf:
        writer = csv.writer(mf)
        if mode == "w":
            writer.writerow(header)

        # 2. Preload static layers
        logging.info("Clipping DEM & computing slope/aspect ...")
        dem = ee.Image("USGS/SRTMGL1_003").clip(region)
        elev = dem.rename("elevation")
        # slope = ee.Terrain.slope(dem).rename("slope")
        # aspect = ee.Terrain.aspect(dem).rename("aspect")

        # 3. Figure out date range
        first_ts = collection.first().get("system:time_start")
        first_date = ee.Date(first_ts).format("YYYY-MM-dd").getInfo()
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # 4. Pull features once
        feats = collection \
            .filterDate(first_date, end_date) \
            .select(["B2","B3","B4","B8"]) \
            .getInfo()["features"]

        # 5. Loop dynamic images
        total = len(feats)
        logging.info(f"Fetching {total} total images, date range {first_date} → {end_date}")
        for idx, feat in enumerate(feats, start=1):
            props = feat["properties"]
            ts    = props["system:time_start"]
            dt    = datetime.fromtimestamp(ts/1000, timezone.utc)
            dt_str = dt.strftime("%Y%m%d")
            sys_idx = props["system:index"]
            datatake = props.get("DATATAKE_IDENTIFIER", "")
            cloud    = props.get("CLOUDY_PIXEL_PERCENTAGE", "")

            # composite bands (only R, G, B, NDVI, elevation)
            img = ee.Image(feat["id"])
            rgb   = img.select(["B4","B3","B2"]).rename(["R","G","B"])
            ndvi  = img.normalizedDifference(["B8","B4"]).rename("NDVI")
            # comp  = ee.Image.cat([rgb, ndvi, elev, slope, aspect])
            comp5 = ee.Image.cat([rgb, ndvi, elev])

            fname = out_dir / f"composite_{dt_str}.tif"
            logging.info(f"({idx}/{total}: Geohash: {geohash}. Exporting image...")
            geemap.ee_export_image(
                # comp,
                comp5,
                filename=str(fname),
                scale=scale, 
                region=[xmin, ymin, xmax, ymax]
            )

            da8 = enrich_with_slope_aspect(in_tif=fname, transform=transform, out_tif=fname)

            # log stats and record to CSV
            log_raster_stats(da8, f"composite_{dt_str}")
            writer.writerow([
                fname.name,
                dt.isoformat(),
                sys_idx,
                datatake,
                cloud,
                True
            ])

        # 6. Static “once per geohash” block (simplified)
        if not skip_static:
            static_ts = first_date

            # 6a) export only elevation
            elev_tif = out_dir / "elevation.tif"
            geemap.ee_export_image(
                elev,
                filename=str(elev_tif),
                scale=scale,
                region=[xmin, ymin, xmax, ymax]
            )

            # 6b) locally compute & write slope/aspect
            da_elev = rxr.open_rasterio(str(elev_tif), masked=False)
            if "band" in da_elev.dims and da_elev.sizes["band"] == 1:
                da_elev = da_elev.squeeze("band", drop=True)

            # slope_da, aspect_da = compute_slope_aspect(da_elev, transform)
            slope_da, aspect_da = compute_slope_aspect_gdal(da_elev, transform)

            # deal with -9999 value
            aspect_da = aspect_da.where(aspect_da != -9999, np.nan)
            aspect_da.rio.write_nodata(np.nan, inplace=True)

            slope_fn  = out_dir / "slope.tif"
            aspect_fn = out_dir / "aspect.tif"
            slope_da.rio.to_raster(str(slope_fn))
            aspect_da.rio.to_raster(str(aspect_fn))

            # 6c) log stats
            log_raster_stats(slope_da, "static_slope")
            log_raster_stats(aspect_da, "static_aspect")

            # 6d) write their CSV rows
            for fn, layer in [(elev_tif, "elevation"),
                            (slope_fn, "slope"),
                            (aspect_fn, "aspect")]:
                writer.writerow([
                    fn.name,
                    f"{static_ts}T00:00:00Z",
                    "",       # no system index for static
                    layer,
                    "",       # no cloud
                    True
                ])

def create_zarr_from_imetadata(
    metadata_csv: Path,
    image_dir: Path,
    zarr_path: str,
    storage_options: dict,
    batch_size: int = 20,
    chunks: dict = None,
    append: bool = False,
    start_date: datetime.date = None
):
    """
    Reads metadata_csv + GeoTIFFs in image_dir, stitches them into an xarray
    Dataset (with dims time * y * x and variables R,G,B,NDVI,elevation,slope,aspect),
    chunks to `chunks`, and writes a brand-new Zarr at zarr_path on S3.
    """
    # 1) read metadata (and for append: drop any timesteps already in the Zarr)
    df = pd.read_csv(metadata_csv, parse_dates=["datetime"])
    # coerce any ISO-8601 timestamp (with or without 'T') into a tz-aware datetime
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        format="ISO8601"   # let pandas handle the mixed ISO formats
    )

    # -- if we're appending, peek at the existing Zarr time axis and drop duplicates --
    if append:
        # open existing store (if possible)
        fs_exist    = s3fs.S3FileSystem(**storage_options)
        store_exist = fs_exist.get_mapper(zarr_path)
        try:
            existing_ds = xr.open_zarr(
                store_exist,
                consolidated=True,
                storage_options=storage_options
            )
            # find the latest date already in the Zarr
            max_time = pd.to_datetime(existing_ds.time.values).max().date()
            logging.info(f"[append] skipping all rows ≤ {max_time}")
            # drop any rows that are on or before that date
            df = df[df["datetime"].dt.date > max_time]
        except Exception as e:
            logging.warning(f"Could not read existing Zarr; appending all CSV rows: {e}")


    if append and start_date is not None:
        # strip off time-of-day/TZ and compare only dates
        df = df[df["datetime"].dt.date >= start_date]
    df = df[
        ~df["DATATAKE_IDENTIFIER"].isin(["elevation","slope","aspect"])
    ].reset_index(drop=True)
    df['static_included'] = True
    # cols = df.columns.tolist()
    # df.to_csv(metadata_csv, columns=cols, index=False)
    cols = df.columns.tolist()
    # only overwrite the CSV on a full rebuild — on append runs
    # the CSV has already been opened in download_all_imagery_with_metadata
    # with mode="a", so we don’t want to clobber it here
    if not append:
        df.to_csv(metadata_csv, columns=cols, index=False)

    n_batches = math.ceil(len(df) / batch_size)

    # 2) prepare the S3 mapper
    fs    = s3fs.S3FileSystem(**storage_options)
    store = fs.get_mapper(zarr_path)

    if append:
        create_mode = False
    else:
        # fresh build: blow away any old store, then write from scratch
        if fs.exists(zarr_path):
            fs.rm(zarr_path, recursive=True)
        create_mode = True
    template_da = None

    # 3) batch‐wise build & write
    for i in range(n_batches):
        batch = df.iloc[i*batch_size : (i+1)*batch_size]
        das = []

        for row in batch.itertuples(index=False):
            # pick filename + variable name
            fname = image_dir / row.filename
            varname = row.DATATAKE_IDENTIFIER

            da = rxr.open_rasterio(str(fname), masked=False)

            # assign band names
            # this if clause will never fire, only the else will
            if varname in ("elevation","slope","aspect"):
                da = da.assign_coords(band=[varname])
            else:
                da = da.assign_coords(band=["R","G","B","NDVI","elevation","slope","aspect", "aspect_mask"])

            # use first DA as template; reproject others to match
            if template_da is None:
                template_da = da
            else:
                da = da.rio.reproject_match(template_da)

            # split band→variables
            ds = da.to_dataset(dim="band")

            # Make 'time' and actual 1-length dimension
            ts = np.datetime64(pd.to_datetime(row.datetime))
            ds = ds.assign_coords(time=[ts])

            # drop any stray chunk directives
            ds.encoding.pop("chunks", None)

            das.append(ds)

        # concat and sort
        small_ds = xr.concat(das, dim="time").sortby("time")
        desired_order = [
            "R", "G", "B", "NDVI", "elevation", "slope", "aspect", "aspect_mask",
            "delta_t_norm", "doy_norm"
        ]
        # Only keep whichever of those actually made it into this batch:
        ordered_vars = [v for v in desired_order if v in small_ds.data_vars]
        small_ds = small_ds[ordered_vars]

        # drop any existing CF-encoding attrs
        for field in ("_FillValue", "add_offset", "scale_factor"):
            small_ds.encoding.pop(field, None)
            for v in small_ds.data_vars:
                small_ds[v].attrs.pop(field, None)
                small_ds[v].encoding.pop(field, None)
            for c in small_ds.coords:
                small_ds[c].attrs.pop(field, None)
                small_ds[c].encoding.pop(field, None)

        # chunk
        if chunks is None:
            chunks = {"time": 1, "y": 256, "x": 256}
        small_ds = small_ds.chunk(chunks)

        # ——— add time-derived bands ———
        # # compute Δt in days (first step gets 0)
        # times = small_ds.time.values
        # dt_vals = np.diff(times) / np.timedelta64(1, "D")
        # dt_vals = np.insert(dt_vals, 0, 0.0)                            # shape (T,)
        # # broadcast to (time, y, x)
        # y, x = small_ds.sizes["y"], small_ds.sizes["x"]
        # dt2d = dt_vals[:, None, None] * np.ones((len(times), y, x))
        # small_ds["delta_t"] = (("time", "y", "x"), dt2d)

        # # day-of-year as fraction of year
        # doys = np.array([pd.to_datetime(t).dayofyear / 365.0 for t in times])
        # doy2d = doys[:, None, None] * np.ones((len(times), y, x))
        # small_ds["doy"] = (("time", "y", "x"), doy2d)

        # # month as integer 1–12 (or scale to [0,1] if you prefer)
        # months = np.array([pd.to_datetime(t).month for t in times])
        # mon2d = months[:, None, None] * np.ones((len(times), y, x))
        # small_ds["month"] = (("time", "y", "x"), mon2d)
        # ——— add time-derived bands and ensure normalized bands every time slice ———
        T = small_ds.sizes["time"]  # or use your actual time‐dim name
        have_dt   = "delta_t_norm" in small_ds.data_vars \
                    and small_ds["delta_t_norm"].sizes["time"] == T
        have_doy  = "doy_norm"      in small_ds.data_vars \
                    and small_ds["doy_norm"].sizes["time"]      == T

        if not (have_dt and have_doy):
            # recompute (or compute for the first time) exactly as before…
            times = small_ds.time.values

            # 1) Raw Δt in days → normalize to [0,1]
            dt_vals = np.diff(times) / np.timedelta64(1, "D")
            dt_vals = np.insert(dt_vals, 0, 0.0)
            max_dt   = float(dt_vals.max()) if dt_vals.max() != 0 else 1.0
            dt_norm  = (dt_vals / max_dt).astype(np.float32)

            # 2) Broadcast into 3D and assign
            y, x = small_ds.sizes["y"], small_ds.sizes["x"]
            dt2d = dt_norm[:, None, None] * np.ones((T, y, x), dtype=np.float32)
            small_ds["delta_t_norm"] = (("time","y","x"), dt2d)

            # 3) Day-of-year as fraction [0,1)
            doys = np.array([pd.to_datetime(t).dayofyear / 365.0 for t in times], dtype=np.float32)
            doy2d = doys[:, None, None] * np.ones((T, y, x), dtype=np.float32)
            small_ds["doy_norm"] = (("time","y","x"), doy2d)

        else:
            print(f"delta_t_norm and doy_norm already exist with length={T}; skipping.")
        # ------

        if create_mode:
            enc = {
                v: {"chunks": tuple(chunks[d] for d in small_ds[v].dims)}
                for v in small_ds.data_vars
            }
            small_ds.to_zarr(
                store,
                mode="w",
                encoding=enc,
                consolidated=False
            )
            create_mode = False
        else:
            small_ds.to_zarr(
                store,
                mode="a",
                append_dim="time",
                consolidated=False
            )
        logging.info(
            f"Wrote Zarr at {zarr_path} with "
            f"{small_ds.sizes['time']} timesteps, "
            f"{small_ds.sizes['y']} * {small_ds.sizes['x']} spatial grid."
        )

        # cleanup
        del das, small_ds
        gc.collect()

    # 4) consolidate
    zarr.convenience.consolidate_metadata(store)

def build_zarr_from_imetadata(
    metadata_csv: Path,
    image_dir: Path,
    zarr_path: str,
    storage_options: dict,
    batch_size: int = 20
):
    """
    Reads your metadata CSV in tmpdir_path, batches the TIFFs into
    small xarray datasets, and writes/appends them to the Zarr at zarr_path.
    """
    # 1) Load metadata, drop static entries
    meta_df = pd.read_csv(metadata_csv, parse_dates=["datetime"])
    n = len(meta_df)
    n_batches = math.ceil(n / batch_size)

    # 2) Prepare S3 store
    fs = s3fs.S3FileSystem(**storage_options)
    store = fs.get_mapper(zarr_path)
    create_mode = not fs.exists(zarr_path)

    template_da = None

    # 3) Iterate batches
    for i in range(n_batches):
        batch = meta_df.iloc[i*batch_size : (i+1)*batch_size]
        das = []

        for row in batch.itertuples(index=False):
            # pick off filename & varname
            if row.DATATAKE_IDENTIFIER in ("elevation","slope","aspect"):
                tif = image_dir / f"{row.DATATAKE_IDENTIFIER}.tif"
                varname = row.DATATAKE_IDENTIFIER
            else:
                # ensure we have a Timestamp, not a raw str
                ts = pd.to_datetime(row.datetime)
                dt = ts.strftime("%Y%m%d")
                tif = image_dir / f"composite_{dt}.tif"
                varname = row.DATATAKE_IDENTIFIER

            da = rxr.open_rasterio(str(tif), masked=True)
            # Make sure we preserve the band dim for every TIFF:
            #  - for composites, length=7  
            #  - for static (elevation/slope/aspect), length=1

            # assign meaningful band names
            if varname in ("elevation","slope","aspect"):
                da = da.assign_coords(band=[ varname ])
            else:
                da = da.assign_coords(band=[
                    "R","G","B","NDVI","elevation","slope","aspect"
                ])

            # split out each band into its own variable
            ds = da.to_dataset(dim="band")
            # now every ds has no 'band' dim—just variables with dims (y,x)

            # on first raster, capture the template grid
            if template_da is None:
                template_da = da
            else:
                # snap to the template's exact grid
                for v in ds.data_vars:
                    ds[v] = ds[v].rio.reproject_match(template_da)
                
            # add time coordinate (this will create the time dimension at concat)
            ts = pd.to_datetime(row.datetime)
            ds = ds.assign_coords(time=np.datetime64(ts))
            ds.encoding.pop("chunks", None)
            das.append(ds)

        # Now every dataset in `das` has only (y,x) dims, the same set of variables,
        # so concat on a new "time" dim will work cleanly:
        small_ds = xr.concat(das, dim="time").sortby("time")

        # drop any stray FillValue so xarray won’t conflict on encoding
        small_ds.encoding.pop("_FillValue", None)
        for var in small_ds.data_vars:
            small_ds[var].encoding.pop("_FillValue", None)
            small_ds[var].attrs   .pop("_FillValue", None)
        for coord in small_ds.coords:
            small_ds[coord].encoding.pop("_FillValue", None)
            small_ds[coord].attrs   .pop("_FillValue", None)

        # define chunks
        chunks = {"time": 1, "y": 256, "x": 256}
        small_ds = small_ds.chunk(chunks)

        if create_mode:
            small_ds.to_zarr(store, 
                             mode="w", 
                             encoding={var: {"chunks": tuple(chunks[d] for d in small_ds[var].dims)}
                                      for var in small_ds.data_vars},
                             consolidated=False)
            create_mode = False
        else:
            small_ds.to_zarr(
                store,
                mode="a",
                append_dim="time",
                consolidated=False
            )

        # free memory
        del das, small_ds
        gc.collect()

    # 4) Consolidate once at the end
    zarr.convenience.consolidate_metadata(store)

def compute_band_stats(
    ds: xr.Dataset,
    bands: list[str],
    stats_path: str,
    storage_options: dict | None = None
):
    """
    Computes per-band mean, std, min, and max across all dims and writes JSON.
    If stats_path starts with "s3://", will write to S3; otherwise to local disk.
    Skips computation if the file already exists.
    
    Args:
        ds (xr.Dataset): input dataset containing the bands
        bands (list[str]): which data_vars to compute stats for
        stats_path (str): path to output JSON, e.g. 
            - "data/gh/band_stats.json"
            - "s3://bucket/data/gh/band_stats.json"
        aws_creds (dict, optional): {"access_key":..., "secret_access_key":...}
    """
    # early exit if already exists
    if stats_path.startswith("s3://"):
        so = storage_options
        s3 = fsspec.filesystem("s3", **so)
        if s3.exists(stats_path):
            print(f"Band stats JSON already on S3 at {stats_path}; skipping.")
            return
    else:
        local = Path(stats_path)
        if local.exists():
            print(f"Band stats JSON already at {stats_path}; skipping.")
            return
        local.parent.mkdir(parents=True, exist_ok=True)

    print(f"Computing band stats for {bands} → {stats_path}")

    stats = {}
    for b in bands:
        arr = ds[b]
        vals = arr.values.astype(float)
        stats[b] = {
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals)),
            "min":  float(np.nanmin(vals)),
            "max":  float(np.nanmax(vals)),
        }

    # write JSON
    if stats_path.startswith("s3://"):
        # upload to S3
        with fsspec.open(stats_path, "w", **so) as f:
            json.dump(stats, f, indent=2)
        print(f"Uploaded band stats to {stats_path}")
    else:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved band stats to {stats_path}")

def get_nearest_image(collection: ee.ImageCollection, target_date: str) -> tuple[str, ee.Image]:
    """
    Returns (date, ee.Image) to those found in the Earth Engine Image Collection whose
    system:time_start is closest to the given YYYY-MM-DD string.
    """
    logging.info(f"Finding nearest image to {target_date}")
    target = ee.Date(target_date)

    def score(img):
        """Compute difference in time in seconds"""
        diff = ee.Date(img.get('system:time_start')) \
                 .difference(target, 'second') \
                 .abs()
        return img.set('delta', diff)

    # map scoring function, sort by delta, and grab the first
    closest = ee.Image(collection
                      .map(score)
                      .sort('delta')
                      .first())
    ts = closest.get('system:time_start').getInfo()
    dt = datetime.fromtimestamp(ts/1000, timezone.utc)
    actual_date = dt.date()
    logging.info(f"Closest date found to {target_date}: {actual_date}")
    return actual_date, closest

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build or update an xarray/Zarr datacube per geohash"
    )
    parser.add_argument(
        "--bucket", "-b", required=True,
        help="Name of the S3 bucket (no s3:// prefix), e.g. 'rgc-zarr-store'"
    )
    parser.add_argument(
        "--folder", "-f", required=True,
        help="Top-level folder (prefix) under the bucket, e.g. 'data'"
    )
    parser.add_argument(
        "--ee-account-key", "-K", required=True,
        help="Path to GEE service account key JSON, e.g. secrets/low-res-sat-change-detection-f7e0f971189b.json"
    )
    parser.add_argument(
        "--ee-account-email", "-E", required=True,
        help="GEE service account email, e.g. low-res-sat-change-detection@low-res-sat-change-detection.iam.gserviceaccount.com"
    )
    parser.add_argument(
        "--aws-creds-file", "-A", required=True,
        help="Path to AWS credentials JSON, e.g. secrets/aws_rgc-zarr-store.json"
    )
    parser.add_argument(
        "--geohash", "-g", required=True,
        help="Comma-separated 5-precision geohash(s), e.g. '9vgm0,9vgm1'"
    )
    parser.add_argument(
        "--date0", type=str, required=True,
        help="First target date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--date1", type=str, required=True,
        help="Second target date (YYYY-MM-DD)"
    )
    # optional args, best left to defaults
    parser.add_argument(
        "--cloud-perc", "-c", type=int, default=5,
        help="Cloud percentage filter for imagery to select below this threshold; defaults to 5%%."
    )
    parser.add_argument(
        "--scale", "-s", type=int, required=False, default=10,
        help="(10) Scale for imagery requested from Google Earth Engine"
    )
    parser.add_argument(
        "--end-date", "-e", required=False, default=datetime.today().strftime('%Y-%m-%d'),
        help="(YYYY-MM-DD) Do not use, debugging purposes only"
    )
    return parser.parse_args()

# def run_for_geohash(
#         geohash: str,
#         date0: str,
#         date1: str,
#         cloud_perc: int = 5,
#         scale: int = 10,
#         end_date: str | None = None,
#         bucket: str = 'rgc-zarr-store',
#         folder: str = 'data',
#         ee_key: str = 'secrets/low-res-sat-change-detection-f7e0f971189b.json',
#         ee_email: str = 'low-res-sat-change-detection@low-res-sat-change-detection.iam.gswerviceaccount.com',
#         aws_creds_file: str = 'secrets/aws_rgc-zarr-store.json',
#         progress_callback: Callable[[float], None] | None = None
# ):
#     """
#     Shares the same logic as main(), but accepts arguments directly,
#     rather than via argparse. Returns when done (or raises on error).
#     """
#     # when looping over the two dates:
#     total_steps = 11
#     step = 0

#     # 1) build AWS creds dict
#     storage_options = make_aws_creds_dict(aws_creds_file)
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 2) Authenticate to GEE
#     authenticate_ee(account_key=ee_key, account_email=ee_email)
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     gh = geohash.strip()
#     region = get_ee_region(gh)
#     bbox = create_bbox(gh)

#     # 3) Create local output folder (and metadata CSV path)
#     out_local = Path(f"./{gh}_two_dates")
#     out_local.mkdir(parents=True, exist_ok=True)

#     metadata_csv = out_local / f"metadata_{gh}.csv"
#     header = ["filename", "datetime", "system_index",
#               "DATATAKE_IDENTIFIER", "CLOUDY_PIXEL_PERCENTAGE", "static_included"]

#     # If no metadata CSV exists yet, write the header now
#     if not metadata_csv.exists():
#         with open(metadata_csv, "w", newline="") as mf:
#             writer = csv.writer(mf)
#             writer.writerow(header)
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 4) Load the EE collection once for that geohash
#     coll = get_ee_collection(region=region, cloud_perc=cloud_perc)
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 5) For each of the two dates, find nearest image, composite, export, enrich, and record metadata
#     for target_date in [date0, date1]:
#         # a) find closest image
#         actual_date, image = get_nearest_image(coll, target_date)

#         # b) build a 5-band composite: R, G, B, NDVI, elevation
#         rgb  = image.select(["B4","B3","B2"]).rename(["R","G","B"])
#         ndvi = image.normalizedDifference(["B8","B4"]).rename("NDVI")
#         dem  = ee.Image("USGS/SRTMGL1_003").clip(region).rename("elevation")
#         composite = ee.Image.cat([rgb, ndvi, dem])

#         # c) prepare filenames & local paths
#         date_str = actual_date.strftime("%Y%m%d")
#         tif_name = f"{gh}_{date_str}.tif"
#         local_tif = out_local / tif_name

#         logging.info(f"Exporting composite for {gh} on {actual_date} → {local_tif.name}")
#         geemap.ee_export_image(
#             composite,
#             filename=str(local_tif),
#             scale=scale,
#             region=[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
#         )

#         # d) Build the Affine transform (same as original script)
#         xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
#         transform = Affine(scale, 0, xmin, 0, -scale, ymax)

#         # e) Enrich with slope + aspect (this overwrites the 5-band TIFF with 8 bands)
#         enriched = enrich_with_slope_aspect(in_tif=local_tif, transform=transform, out_tif=local_tif)
#         log_raster_stats(enriched, f"{gh}_{date_str}")

#         # f) Grab the image properties for metadata
#         info = image.getInfo()
#         props = info.get("properties", {})

#         ts = props.get("system:time_start", None)
#         dt_iso = (datetime.fromtimestamp(ts/1000, timezone.utc).isoformat() if ts
#                   else actual_date.isoformat() + "T00:00:00Z")
#         sys_idx   = props.get("system:index", "")
#         datatake  = props.get("DATATAKE_IDENTIFIER", "")
#         cloud_pct = props.get("CLOUDY_PIXEL_PERCENTAGE", "")

#         # g) Append a row to metadata CSV
#         with open(metadata_csv, "a", newline="") as mf:
#             writer = csv.writer(mf)
#             writer.writerow([
#                 tif_name,
#                 dt_iso,
#                 sys_idx,
#                 datatake,
#                 cloud_pct,
#                 True   # static_included = True
#             ])

#         logging.info(f"Wrote metadata row for {tif_name} to {metadata_csv.name}")
        
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 6) Create (or overwrite) a Zarr on S3 using create_zarr_from_imetadata
#     _, zarr_path, _ = build_s3_path(geohash=gh, bucket=bucket, folder=folder)
#     zarr_path = f"{zarr_path.rstrip('/')}_streamlit"
#     logging.info(f"Creating Zarr for {gh} → {zarr_path}")
#     create_zarr_from_imetadata(
#         metadata_csv=metadata_csv,
#         image_dir=out_local,
#         zarr_path=zarr_path,
#         storage_options=storage_options,
#         batch_size=20,
#         chunks={"time": 1, "y": 256, "x": 256},
#         append=False,
#         start_date=None
#     )
#     logging.info(f"Zarr successfully written to {zarr_path}")
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 7) Upload the two TIFFs themselves to S3
#     s3 = s3fs.S3FileSystem(**storage_options)
#     for tif in sorted(out_local.iterdir()):
#         if tif.suffix.lower() == ".tif":
#             remote_tif = f"{bucket}/{folder}/{gh}/{tif.name}"
#             logging.info(f"Uploading {tif.name} → s3://{remote_tif}")
#             s3.put(str(tif), remote_tif)
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 8) Upload the metadata CSV to S3
#     remote_csv = f"{bucket}/{folder}/{gh}/{metadata_csv.name}"
#     logging.info(f"Uploading metadata CSV → s3://{remote_csv}")
#     s3.put(str(metadata_csv), remote_csv)
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 9) Build climatology layers & band_stats.json
#     #    a) Open the newly-written Zarr from S3
#     fs = s3fs.S3FileSystem(**storage_options)
#     mapper = fs.get_mapper(f"{bucket}/{folder}/{gh}/{gh}_zarr")
#     ds = xr.open_zarr(mapper, consolidated=True)
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     #    b) Compute and upload seasonal & monthly NDVI climatology
#     base_clim = f"s3://{bucket}/{folder}/{gh}/{gh}_climatology"
#     logging.info(f"Computing climatology layers → {base_clim}")
#     compute_climatology(
#         ds=ds,
#         save_dir=base_clim,
#         storage_options=storage_options
#     )
#     logging.info(f"Climatology layers written to {base_clim}")
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     #    c) Compute and upload band_stats.json alongside climatology
#     stats_path = f"{base_clim.rstrip('/')}/band_stats.json"
#     logging.info(f"Computing band stats → {stats_path}")
#     compute_band_stats(
#         ds=ds,
#         bands=["R","G","B","NDVI","elevation","slope","aspect","aspect_mask"],
#         stats_path=stats_path,
#         storage_options=storage_options
#     )
#     logging.info(f"Band statistics written to {stats_path}")

#     logging.info(
#         f"All done!\n"
#         f" • Two dates (8-band GeoTIFFs) are in {out_local}/\n"
#         f" • Metadata CSV: {metadata_csv} → uploaded to s3://{remote_csv}\n"
#         f" • Zarr on S3: {zarr_path}\n"
#         f" • Climatology files (seasonal/monthly NDVI) in: {base_clim}\n"
#         f" • Band stats JSON: {stats_path}"
#     )
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)
# def run_for_geohash_local(
#     geohash: str,
#     date0: str,
#     date1: str,
#     cloud_perc: int = 5,
#     scale: int = 10,
#     ee_key: str = "secrets/low-res-sat-change-detection-f7e0f971189b.json",
#     ee_email: str = "low-res-sat-change-detection@low-res-sat-change-detection.iam.gswerviceaccount.com",
#     progress_callback: callable | None = None,
#     local_store_root: str = "./local_store",
# ):
#     """
#     100% local build:
#     1) Authenticate EE
#     2) Download composites for date0/date1
#     3) Enrich TIFFs (slope/aspect)
#     4) Write metadata CSV
#     5) Build a local Zarr store
#     6) Compute climatology locally
#     7) Compute band_stats.json locally
#     """

#     total_steps = 9
#     step = 0
#     gh = geohash.strip()
#     assert len(gh) == 5, "Expected a 5-char geohash"

#     # 1) Authenticate Earth Engine
#     authenticate_ee(account_key=ee_key, account_email=ee_email)
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 2) Build bbox & EE region
#     region = get_ee_region(gh)
#     bbox = create_bbox(gh)

#     # 3) Create local folders
#     root = Path(local_store_root) / gh
#     out_two_dates = root / "two_dates"
#     out_two_dates.mkdir(parents=True, exist_ok=True)

#     metadata_csv = out_two_dates / f"metadata_{gh}.csv"
#     header = [
#         "filename",
#         "datetime",
#         "system_index",
#         "DATATAKE_IDENTIFIER",
#         "CLOUDY_PIXEL_PERCENTAGE",
#         "static_included",
#     ]
#     if not metadata_csv.exists():
#         with open(metadata_csv, "w", newline="") as mf:
#             writer = csv.writer(mf)
#             writer.writerow(header)

#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 4) Load EE collection
#     coll = get_ee_collection(region=region, cloud_perc=cloud_perc)
#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 5) For date0 & date1: download composite, enrich, record metadata
#     for target_date in [date0, date1]:
#         actual_date, image = get_nearest_image(coll, target_date)

#         # Build 5-band composite (R,G,B,NDVI,elevation)
#         rgb = image.select(["B4", "B3", "B2"]).rename(["R", "G", "B"])
#         ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
#         dem = ee.Image("USGS/SRTMGL1_003").clip(region).rename("elevation")
#         composite = ee.Image.cat([rgb, ndvi, dem])

#         date_str = actual_date.strftime("%Y%m%d")
#         tif_name = f"{gh}_{date_str}.tif"
#         local_tif = out_two_dates / tif_name

#         logging.info(f"Exporting composite for {gh} on {actual_date} → {local_tif}")
#         geemap.ee_export_image(
#             composite,
#             filename=str(local_tif),
#             scale=scale,
#             region=[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]],
#         )

#         xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
#         transform = Affine(scale, 0, xmin, 0, -scale, ymax)

#         # Enrich TIFF with slope & aspect
#         enriched = enrich_with_slope_aspect(in_tif=local_tif, transform=transform, out_tif=local_tif)
#         log_raster_stats(enriched, f"{gh}_{date_str}")

#         info = image.getInfo()
#         props = info.get("properties", {})

#         ts = props.get("system:time_start", None)
#         if ts is not None:
#             dt_iso = datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat()
#         else:
#             dt_iso = actual_date.isoformat() + "T00:00:00Z"

#         sys_idx = props.get("system:index", "")
#         datatake = props.get("DATATAKE_IDENTIFIER", "")
#         cloud_pct = props.get("CLOUDY_PIXEL_PERCENTAGE", "")

#         with open(metadata_csv, "a", newline="") as mf:
#             writer = csv.writer(mf)
#             writer.writerow([tif_name, dt_iso, sys_idx, datatake, cloud_pct, True])

#         logging.info(f"Wrote metadata row for {tif_name} to {metadata_csv}")

#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 6) Build a local Zarr store
#     base_local_zarr = root / f"{gh}_zarr_local"
#     if base_local_zarr.exists():
#         shutil.rmtree(base_local_zarr)
#     zarr_path = str(base_local_zarr.resolve())

#     logging.info(f"Creating LOCAL Zarr for {gh} → {zarr_path}")
#     create_zarr_from_imetadata(
#         metadata_csv=metadata_csv,
#         image_dir=out_two_dates,
#         zarr_path=zarr_path,
#         storage_options=None,  # purely local
#         batch_size=20,
#         chunks={"time": 1, "y": 256, "x": 256},
#         append=False,
#         start_date=None,
#     )
#     logging.info(f"Local Zarr successfully written to {zarr_path}")

#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 7) Compute & write local climatology
#     base_clim = (root / f"{gh}_climatology_local").as_posix()
#     logging.info(f"Computing LOCAL climatology layers → {base_clim}")

#     ds_zarr = xr.open_zarr(str(Path(local_store_root) / gh / f"{gh}_zarr_local"))
#     compute_climatology(
#         ds=ds_zarr,
#         save_dir=base_clim,
#         storage_options=None,
#     )
#     logging.info(f"Local climatology layers written to {base_clim}")

#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 8) Compute & write local band_stats.json
#     ds_for_stats = xr.open_zarr(str(Path(local_store_root) / gh / f"{gh}_zarr_local"))
#     stats_path = str(Path(local_store_root) / gh / f"{gh}_climatology_local" / "band_stats.json")

#     logging.info(f"Computing LOCAL band stats → {stats_path}")
#     compute_band_stats(
#         ds=ds_for_stats,
#         bands=["R", "G", "B", "NDVI", "elevation", "slope", "aspect", "aspect_mask"],
#         stats_path=stats_path,
#         storage_options=None,
#     )
#     logging.info(f"Local band_stats.json written to {stats_path}")

#     step += 1
#     if progress_callback:
#         progress_callback(step / total_steps)

#     # 9) Final logging & return
#     logging.info(
#         f"LOCAL mode: Finished building everything under {root}\n"
#         f" • Two dates (8-band GeoTIFFs) in {out_two_dates}/\n"
#         f" • Metadata CSV: {metadata_csv}\n"
#         f" • Local Zarr: {base_local_zarr}\n"
#         f" • Climatology files in: {base_clim}\n"
#         f" • Band stats JSON: {stats_path}\n"
#     )

#     return {
#         "tif0": str((out_two_dates / f"{gh}_{date0.replace('-','')}.tif").resolve()),
#         "tif1": str((out_two_dates / f"{gh}_{date1.replace('-','')}.tif").resolve()),
#         "metadata_csv": str(metadata_csv.resolve()),
#         "zarr_path": zarr_path,
#         "climatology_dir": base_clim,
#         "band_stats": stats_path,
#     }
def run_for_geohash(
    geohash: str,
    date0: str,
    date1: str,
    cloud_perc: int = 5,
    scale: int = 10,
    end_date: str | None = None,
    bucket: str = "rgc-zarr-store",
    folder: str = "data",
    ee_key: str = "secrets/low-res-sat-change-detection-f7e0f971189b.json",
    ee_email: str = "low-res-sat-change-detection@low-res-sat-change-detection.iam.gswerviceaccount.com",
    aws_creds_file: str = "secrets/aws_rgc-zarr-store.json",
    progress_callback: Callable | None = None
):
    """
    S3-enabled pipeline for building a datacube for one geohash & two dates.
    1) Read AWS creds  → storage_options (never None)
    2) Authenticate Earth Engine
    3) Download two composites, enrich them locally, write metadata CSV
    4) Build a Zarr on S3 (via create_zarr_from_imetadata)
    5) Upload TIFFs + metadata CSV to S3
    6) Compute climatology on S3
    7) Compute band_stats.json on S3
    """

    total_steps = 10
    step = 0

    gh = geohash.strip()
    if len(gh) != 5:
        raise ValueError(f"Expected a 5‐character geohash, got '{geohash}'")

    # ─── 1) Load AWS creds dict ────────────────────────────────────────────────
    storage_options = make_aws_creds_dict(aws_creds_file)
    if not isinstance(storage_options, dict):
        raise RuntimeError("make_aws_creds_dict did not return a dict")
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 2) Authenticate Earth Engine ─────────────────────────────────────────
    authenticate_ee(account_key=ee_key, account_email=ee_email)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 3) Build bbox & region for the geohash ───────────────────────────────
    region = get_ee_region(gh)
    bbox = create_bbox(gh)

    # ─── 4) Create a local "two_dates" folder ─────────────────────────────────
    out_local = Path(f"./{gh}_two_dates")
    out_local.mkdir(parents=True, exist_ok=True)

    metadata_csv = out_local / f"metadata_{gh}.csv"
    header = [
        "filename",
        "datetime",
        "system_index",
        "DATATAKE_IDENTIFIER",
        "CLOUDY_PIXEL_PERCENTAGE",
        "static_included",
    ]
    if not metadata_csv.exists():
        with open(metadata_csv, "w", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow(header)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 5) Load EE collection for that geohash ─────────────────────────────────
    coll = get_ee_collection(region=region, cloud_perc=cloud_perc)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 6) For each of the two dates: download composite, enrich, record metadata ─
    for target_date in [date0, date1]:
        # a) Find the nearest image in EE
        actual_date, image = get_nearest_image(coll, target_date)

        # b) Build a 5-band composite: (R, G, B, NDVI, elevation)
        rgb = image.select(["B4", "B3", "B2"]).rename(["R", "G", "B"])
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        dem = ee.Image("USGS/SRTMGL1_003").clip(region).rename("elevation")
        composite = ee.Image.cat([rgb, ndvi, dem])

        # c) Prepare local TIFF path
        date_str = actual_date.strftime("%Y%m%d")
        tif_name = f"{gh}_{date_str}.tif"
        local_tif = out_local / tif_name

        logging.info(f"Exporting composite for {gh} on {actual_date} → {local_tif}")
        geemap.ee_export_image(
            composite,
            filename=str(local_tif),
            scale=scale,
            region=[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]],
        )

        # d) Build Affine transform for slope/aspect enrichment
        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        transform = Affine(scale, 0, xmin, 0, -scale, ymax)

        # e) Enrich TIFF with slope & aspect (overwrites the same file)
        enriched = enrich_with_slope_aspect(in_tif=local_tif, transform=transform, out_tif=local_tif)
        log_raster_stats(enriched, f"{gh}_{date_str}")

        # f) Grab image properties & append to metadata CSV
        info = image.getInfo()
        props = info.get("properties", {})

        ts = props.get("system:time_start", None)
        if ts is not None:
            dt_iso = datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat()
        else:
            dt_iso = actual_date.isoformat() + "T00:00:00Z"

        sys_idx = props.get("system:index", "")
        datatake = props.get("DATATAKE_IDENTIFIER", "")
        cloud_pct = props.get("CLOUDY_PIXEL_PERCENTAGE", "")

        with open(metadata_csv, "a", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow([tif_name, dt_iso, sys_idx, datatake, cloud_pct, True])

        logging.info(f"Wrote metadata row for {tif_name} to {metadata_csv}")

    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 7) Build a Zarr on S3 from the local CSV + TIFFs ────────────────────────
    # Note: build_s3_path() constructs something like “s3://bucket/folder/gh/gh_zarr”
    from build_dataset_streamlit import build_s3_path

    _, zarr_s3_path, _ = build_s3_path(geohash=gh, bucket=bucket, folder=folder)
    zarr_s3_path = f"{zarr_s3_path.rstrip('/')}_streamlit"
    logging.info(f"Creating S3 Zarr for {gh} → {zarr_s3_path}")
    create_zarr_from_imetadata(
        metadata_csv=metadata_csv,
        image_dir=out_local,
        zarr_path=zarr_s3_path,
        storage_options=storage_options,  # guaranteed non-None
        batch_size=20,
        chunks={"time": 1, "y": 256, "x": 256},
        append=False,
        start_date=None,
    )
    logging.info(f"S3 Zarr successfully written to {zarr_s3_path}")

    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 8) Upload the two enriched TIFFs themselves to S3 ───────────────────────
    s3 = s3fs.S3FileSystem(**storage_options)
    for tif in sorted(out_local.iterdir()):
        if tif.suffix.lower() == ".tif":
            remote_tif = f"{bucket}/{folder}/{gh}/{tif.name}"
            logging.info(f"Uploading {tif.name} → s3://{remote_tif}")
            s3.put(str(tif), remote_tif)

    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 9) Upload the metadata CSV to S3 ───────────────────────────────────────
    remote_csv = f"{bucket}/{folder}/{gh}/{metadata_csv.name}"
    logging.info(f"Uploading metadata CSV → s3://{remote_csv}")
    s3.put(str(metadata_csv), remote_csv)

    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 10) Build climatology layers on S3 ─────────────────────────────────────
    base_clim = f"s3://{bucket}/{folder}/{gh}/{gh}_climatology"
    logging.info(f"Computing S3 climatology layers → {base_clim}")
    # We must open the Zarr we just wrote
    ds_for_clim = xr.open_zarr(
        s3fs.S3FileSystem(**storage_options).get_mapper(zarr_s3_path),
        consolidated=True,
        storage_options=storage_options,
    )
    compute_climatology(
        ds=ds_for_clim,
        save_dir=base_clim,
        storage_options=storage_options,
    )
    logging.info(f"S3 climatology layers written to {base_clim}")

    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── 11) Build band_stats.json on S3 ───────────────────────────────────────
    stats_path = f"{base_clim.rstrip('/')}/band_stats.json"
    logging.info(f"Computing S3 band stats → {stats_path}")
    compute_band_stats(
        ds=ds_for_clim,
        bands=["R", "G", "B", "NDVI", "elevation", "slope", "aspect", "aspect_mask"],
        stats_path=stats_path,
        storage_options=storage_options,
    )
    logging.info(f"S3 band_stats.json written to {stats_path}")

    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # ─── Final logging ──────────────────────────────────────────────────────────
    logging.info(
        f"S3 mode: Finished building & uploading everything:\n"
        f" • Two dates (8‐band GeoTIFFs) in {out_local}/ (also uploaded to S3)\n"
        f" • Metadata CSV: {metadata_csv} → s3://{bucket}/{folder}/{gh}/{metadata_csv.name}\n"
        f" • S3 Zarr: {zarr_s3_path}\n"
        f" • Climatology files (seasonal/monthly NDVI) in: {base_clim}\n"
        f" • Band stats JSON: {stats_path}\n"
    )

    return {
        "local_tif0": str((out_local / f"{gh}_{date0.replace('-', '')}.tif").resolve()),
        "local_tif1": str((out_local / f"{gh}_{date1.replace('-', '')}.tif").resolve()),
        "metadata_csv": str(metadata_csv.resolve()),
        "zarr_s3_uri": zarr_s3_path,
        "climatology_s3_dir": base_clim,
        "band_stats_s3": stats_path,
    }

def main():
    args = parse_args()

    # 1) build AWS creds dict
    storage_options = make_aws_creds_dict(args.aws_creds_file)

    # 2) Authenticate to GEE
    authenticate_ee(account_key=args.ee_account_key, account_email=args.ee_account_email)

    gh = args.geohash.strip()
    region = get_ee_region(gh)
    bbox = create_bbox(gh)

    # 3) Create local output folder (and metadata CSV path)
    out_local = Path(f"./{gh}_two_dates")
    out_local.mkdir(parents=True, exist_ok=True)

    metadata_csv = out_local / f"metadata_{gh}.csv"
    header = ["filename", "datetime", "system_index",
              "DATATAKE_IDENTIFIER", "CLOUDY_PIXEL_PERCENTAGE", "static_included"]

    # If no metadata CSV exists yet, write the header now
    if not metadata_csv.exists():
        with open(metadata_csv, "w", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow(header)

    # 4) Load the EE collection once for that geohash
    coll = get_ee_collection(region=region, cloud_perc=args.cloud_perc)

    # 5) For each of the two dates, find nearest image, composite, export, enrich, and record metadata
    for target_date in [args.date0, args.date1]:
        # a) find closest image
        actual_date, image = get_nearest_image(coll, target_date)

        # b) build a 5-band composite: R, G, B, NDVI, elevation
        rgb  = image.select(["B4","B3","B2"]).rename(["R","G","B"])
        ndvi = image.normalizedDifference(["B8","B4"]).rename("NDVI")
        dem  = ee.Image("USGS/SRTMGL1_003").clip(region).rename("elevation")
        composite = ee.Image.cat([rgb, ndvi, dem])

        # c) prepare filenames & local paths
        date_str = actual_date.strftime("%Y%m%d")
        tif_name = f"{gh}_{date_str}.tif"
        local_tif = out_local / tif_name

        logging.info(f"Exporting composite for {gh} on {actual_date} → {local_tif.name}")
        geemap.ee_export_image(
            composite,
            filename=str(local_tif),
            scale=args.scale,
            region=[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
        )

        # d) Build the Affine transform (same as original script)
        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        transform = Affine(args.scale, 0, xmin, 0, -args.scale, ymax)

        # e) Enrich with slope + aspect (this overwrites the 5-band TIFF with 8 bands)
        enriched = enrich_with_slope_aspect(in_tif=local_tif, transform=transform, out_tif=local_tif)
        log_raster_stats(enriched, f"{gh}_{date_str}")

        # f) Grab the image properties for metadata
        info = image.getInfo()
        props = info.get("properties", {})

        ts = props.get("system:time_start", None)
        if ts is not None:
            dt_full = datetime.fromtimestamp(ts/1000, timezone.utc)
            dt_iso = dt_full.isoformat()
        else:
            dt_iso = actual_date.isoformat() + "T00:00:00Z"

        sys_idx   = props.get("system:index", "")
        datatake  = props.get("DATATAKE_IDENTIFIER", "")
        cloud_pct = props.get("CLOUDY_PIXEL_PERCENTAGE", "")

        # g) Append a row to metadata CSV
        with open(metadata_csv, "a", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow([
                tif_name,
                dt_iso,
                sys_idx,
                datatake,
                cloud_pct,
                True   # static_included = True
            ])

        logging.info(f"Wrote metadata row for {tif_name} to {metadata_csv.name}")

    # 6) Create (or overwrite) a Zarr on S3 using create_zarr_from_imetadata
    _, zarr_path, _ = build_s3_path(geohash=gh, bucket=args.bucket, folder=args.folder)
    logging.info(f"Creating Zarr for {gh} → {zarr_path}")
    create_zarr_from_imetadata(
        metadata_csv=metadata_csv,
        image_dir=out_local,
        zarr_path=zarr_path,
        storage_options=storage_options,
        batch_size=20,
        chunks={"time": 1, "y": 256, "x": 256},
        append=False,
        start_date=None
    )
    logging.info(f"Zarr successfully written to {zarr_path}")

    # 7) Upload the two TIFFs themselves to S3
    s3 = s3fs.S3FileSystem(**storage_options)
    for tif in sorted(out_local.iterdir()):
        if tif.suffix.lower() == ".tif":
            remote_tif = f"{args.bucket}/{args.folder}/{gh}/{tif.name}"
            logging.info(f"Uploading {tif.name} → s3://{remote_tif}")
            s3.put(str(tif), remote_tif)

    # 8) Upload the metadata CSV to S3
    remote_csv = f"{args.bucket}/{args.folder}/{gh}/{metadata_csv.name}"
    logging.info(f"Uploading metadata CSV → s3://{remote_csv}")
    s3.put(str(metadata_csv), remote_csv)

    # 9) Build climatology layers & band_stats.json
    #    a) Open the newly-written Zarr from S3
    fs = s3fs.S3FileSystem(**storage_options)
    mapper = fs.get_mapper(f"{args.bucket}/{args.folder}/{gh}/{gh}_zarr")
    ds = xr.open_zarr(mapper, consolidated=True)

    #    b) Compute and upload seasonal & monthly NDVI climatology
    base_clim = f"s3://{args.bucket}/{args.folder}/{gh}/{gh}_climatology"
    logging.info(f"Computing climatology layers → {base_clim}")
    compute_climatology(
        ds=ds,
        save_dir=base_clim,
        storage_options=storage_options
    )
    logging.info(f"Climatology layers written to {base_clim}")

    #    c) Compute and upload band_stats.json alongside climatology
    stats_path = f"{base_clim.rstrip('/')}/band_stats.json"
    logging.info(f"Computing band stats → {stats_path}")
    compute_band_stats(
        ds=ds,
        bands=["R","G","B","NDVI","elevation","slope","aspect","aspect_mask"],
        stats_path=stats_path,
        storage_options=storage_options
    )
    logging.info(f"Band statistics written to {stats_path}")

    print(
        f"All done!\n"
        f" • Two dates (8-band GeoTIFFs) are in {out_local}/\n"
        f" • Metadata CSV: {metadata_csv} → uploaded to s3://{remote_csv}\n"
        f" • Zarr on S3: {zarr_path}\n"
        f" • Climatology files (seasonal/monthly NDVI) in: {base_clim}\n"
        f" • Band stats JSON: {stats_path}"
    )

if __name__ == "__main__":
    main()
# Example:
# python build_dataset.py \
#   --geohashes 9vgm0,9vgm1 \
#   --bucket rgc-zarr-store \
#   --folder data
#   --start-date 1900-01-01 \
#   --ee-service-account-json secrets/low-res-sat-change-detection-f7e0f971189b.json \
#   --aws-creds-file secrets/aws.json \

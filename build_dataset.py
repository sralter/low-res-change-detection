"""
build_dataset.py

Utilities for downloading Sentinel-2 imagery via Google Earth Engine,
computing static layers (slope, aspect, climatologies), and building
or updating an xarray/Zarr datacube per geohash on S3.

Main features:
  - Authenticate to GEE and AWS
  - Build bounding boxes from geohashes
  - Download imagery & enrich with NDVI, slope, aspect
  - Stitch GeoTIFFs into Zarr stores (initial or incremental)
  - Compute seasonal/monthly NDVI climatologies & per-band stats
"""

import argparse
import json
from datetime import datetime, timezone, timedelta
import geohash2
import ee, geemap
from pathlib import Path
import tempfile
import csv
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

# Reduce noise from dependencies
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("aiobotocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("s3fs").setLevel(logging.WARNING)
logging.getLogger("rasterio").setLevel(logging.WARNING)

def authenticate_ee(account_key: str = None, account_email: str = None):
    """
    Authenticate with Google Earth Engine (GEE).

    If both `account_key` and `account_email` are provided, uses a
    service account; otherwise falls back to default/interactive login.

    Args:
        account_key: Path to GEE service account JSON key.
        account_email: Service account email address.

    Raises:
        FileNotFoundError: If `account_key` is provided but the file
            does not exist.
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
    """
    Compute a lat/lon bounding box from a 5-character geohash.

    Args:
        geohash: 5-character geohash string.

    Returns:
        A dict with keys 'xmin','ymin','xmax','ymax' for the box.
    """
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
    """
    Wrap a basic bbox dict into an EarthEngine Geometry.BBox.

    Args:
        bbox: dict with 'xmin','ymin','xmax','ymax'.

    Returns:
        ee.Geometry.BBox over that region.
    """
    logging.info(f"Creating Earth Engine-friendly bounding box")
    region = ee.Geometry.BBox(
        bbox['xmin'], bbox['ymin'],
        bbox['xmax'], bbox['ymax']
    )
    return region

def get_ee_region(geohash: str) -> ee.Geometry.BBox:
    """
    Get an EE Geometry.BBox directly from a geohash.

    Combines `create_bbox` + `create_ee_bbox`.

    Args:
        geohash: 5-character geohash string.

    Returns:
        EE BBox for that cell.
    """
    bbox = create_bbox(geohash)
    region = create_ee_bbox(bbox)
    return region

def get_ee_collection(region: ee.Geometry.BBox, 
                      cloud_perc: int) -> ee.ImageCollection:
    """
    Fetch a Sentinel-2 surface reflectance collection over a region.

    Args:
        region: EE BBox to filter on.
        cloud_perc: Max CLOUDY_PIXEL_PERCENTAGE to include.

    Returns:
        Filtered ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').
    """
    logging.info(f"Getting Earth Engine Image Collection with cloudy percentage below {cloud_perc}")
    coll_all = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))
    return coll_all

def build_s3_path(geohash: str, bucket: str, folder: str) -> str:
    """
    Construct S3 URIs for storing Zarr and climatology.

    Args:
        geohash: 5-character geohash.
        bucket: S3 bucket name (no 's3://').
        folder: Prefix folder in bucket.

    Returns:
        Tuple: (base_uri, zarr_path, climatology_path).
    """
    bucket = bucket.strip("/")
    folder = folder.strip("/")
    base_uri = f"s3://{bucket}/{folder}"
    geohash_folder = f"{base_uri}/{geohash}"
    zarr_path = f"{geohash_folder}/{geohash}_zarr/"
    climatology_path = f"{geohash_folder}/{geohash}_climatology"
    return base_uri, zarr_path, climatology_path

def make_aws_creds_dict(aws_secrets_file: str) -> dict:
    """
    Load AWS access keys from JSON into storage_options for fsspec/S3FS.

    Args:
        aws_secrets_file: Path to JSON with 'access_key' & 'secret_access_key'.

    Returns:
        dict with keys 'key' and 'secret' for S3FS.
    """
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
    """
    Check whether the Zarr store already exists in S3.

    Args:
        geohash: 5-character geohash.
        bucket: S3 bucket name.
        folder: Prefix folder.
        storage_options: AWS creds for S3FS.

    Returns:
        (exists, zarr_path)
    """
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
                        geohash: str) -> tuple[np.datetime64,np.datetime64]:
    """
    Inspect an existing Zarr on S3 and return its min/max dates.

    Args:
        zarr_path: S3 URI to the Zarr store.
        storage_options: AWS creds for S3FS.
        geohash: Geohash (used for logging).

    Returns:
        (start_date, end_date) as numpy datetime64[D].
    """
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

def compute_slope_aspect_gdal(
    elev_da: xr.DataArray,
    transform: Affine,
    crs_epsg: int = 4326
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Use GDAL DEMProcessing to compute slope and aspect.

    Args:
        elev_da: Elevation DataArray.
        transform: Affine geo-transform.
        crs_epsg: EPSG code for projection (default 4326).

    Returns:
        (slope_da, aspect_da) as DataArrays.
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
    Build and save NDVI climatology (seasonal & monthly) to netCDF.

    Args:
        ds: Input xarray Dataset with 'ndvi' or 'NDVI'.
        save_dir: Local or "s3://…" path to output folder.
        storage_options: AWS creds for S3 writes (if S3).

    Side effects:
        Writes `seasonal_ndvi.nc`, `monthly_ndvi_mean.nc`,
        and `monthly_ndvi_std.nc` under `save_dir`.
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
    Read a 5-band GeoTIFF (R,G,B,NDVI,elev), compute slope/aspect,
    append an aspect_mask band, and write out an 8-band TIFF.

    Args:
        in_tif: Input 5-band GeoTIFF path.
        transform: Affine transform for the raster.
        out_tif: Path to overwrite with 8-band output.

    Returns:
        xr.DataArray of the enriched 8-band raster.
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
    """
    Log basic statistics (min, max, valid count) of a DataArray.

    Args:
        da: DataArray to inspect.
        label: Label for logging.
    """
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
    Download every image in `collection`, enrich to R,G,B,NDVI,elev,slope,aspect,
    write per-date GeoTIFFs under `out_dir`, and append metadata rows to CSV.

    Args:
        collection: EE ImageCollection to iterate.
        out_dir: Local folder for output TIFFs.
        bbox: Dict bounding box for logging.
        scale: Export resolution.
        region: EE BBox geometry.
        metadata_csv: Path to CSV to append rows.
        geohash: Geohash (for logging).
        end_date: Last date to include (defaults to today).
        skip_static: If True, do not re-export static layers.
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
    Stitch TIFFs listed in `metadata_csv` into a Zarr store.

    Reads the CSV, loads each GeoTIFF, converts to xarray Dataset,
    appends or overwrites at `zarr_path`, and consolidates metadata.

    Args:
        metadata_csv: CSV with filename + DATATAKE_IDENTIFIER rows.
        image_dir: Directory where TIFFs live.
        zarr_path: S3 URI or local path for Zarr store.
        storage_options: AWS creds for S3 writes.
        batch_size: Number of timesteps per write batch.
        chunks: Chunk dims for the Zarr variables.
        append: If True, append to existing Zarr; otherwise overwrite.
        start_date: If appending, drop CSV rows ≤ this date.
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

def get_nearest_image(collection: ee.ImageCollection, 
                      target_date: str) -> tuple[str, ee.Image]:
    """
    In an EE collection, find the image whose timestamp is closest
    to `target_date` (YYYY-MM-DD).

    Args:
        collection: EE ImageCollection.
        target_date: Date string "YYYY-MM-DD".

    Returns:
        (actual_date, ee.Image) pair of the closest match.
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
    """
    Parse command-line arguments for build_dataset.py:

      --bucket (-b):   S3 bucket name (no s3://)
      --folder (-f):   Prefix under bucket
      --ee-account-key: GEE service account JSON path
      --ee-account-email: Service account email
      --aws-creds-file: AWS creds JSON path
      --geohashes (-g): Comma-separated geohash list
      --cloud-perc (-c): Cloud filter pct
      --scale (-s):     GEE export scale
      --end-date (-e):  (debug only) override end date

    Returns:
        argparse.Namespace
    """
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
        "--geohashes", "-g", required=True,
        help="Comma-separated 5-precision geohash(s), e.g. '9vgm0,9vgm1'"
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

def run_for_geohash(
        geohash: str,
        date0: str,
        date1: str,
        cloud_perc: int = 5,
        scale: int = 10,
        end_date: str | None = None,
        bucket: str = 'rgc-zarr-store',
        folder: str = 'data',
        ee_key: str = 'secrets/low-res-sat-change-detection-f7e0f971189b.json',
        ee_email: str = 'low-res-sat-change-detection@low-res-sat-change-detection.iam.gswerviceaccount.com',
        aws_creds_file: str = 'secrets/aws_rgc-zarr-store.json',
        progress_callback: Callable[[float], None] | None = None
):
    """
    Shares the same logic as main(), but accepts arguments directly,
    rather than via argparse. Returns when done (or raises on error).
    Used only in the Streamlit tool.
    """
    # when looping over the two dates:
    total_steps = 2 + 1 + 1 + 1  # (2 date composites) + (Zarr build) + (upload) + (climatology)
    step = 0

    # 1) build AWS creds dict
    storage_options = make_aws_creds_dict(aws_creds_file)

    # 2) Authenticate to GEE
    authenticate_ee(account_key=ee_key, account_email=ee_email)

    gh = geohash.strip()
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
    for target_date in [date0, date1]:
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
            scale=scale,
            region=[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
        )

        # d) Build the Affine transform (same as original script)
        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        transform = Affine(scale, 0, xmin, 0, -scale, ymax)

        # e) Enrich with slope + aspect (this overwrites the 5-band TIFF with 8 bands)
        enriched = enrich_with_slope_aspect(in_tif=local_tif, transform=transform, out_tif=local_tif)
        log_raster_stats(enriched, f"{gh}_{date_str}")

        # f) Grab the image properties for metadata
        info = image.getInfo()
        props = info.get("properties", {})

        ts = props.get("system:time_start", None)
        dt_iso = (datetime.fromtimestamp(ts/1000, timezone.utc).isoformat() if ts
                  else actual_date.isoformat() + "T00:00:00Z")
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
        step += 1
        if progress_callback:
            progress_callback(step / total_steps)

    # 6) Create (or overwrite) a Zarr on S3 using create_zarr_from_imetadata
    _, zarr_path, _ = build_s3_path(geohash=gh, bucket=bucket, folder=folder)
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
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # 7) Upload the two TIFFs themselves to S3
    s3 = s3fs.S3FileSystem(**storage_options)
    for tif in sorted(out_local.iterdir()):
        if tif.suffix.lower() == ".tif":
            remote_tif = f"{bucket}/{folder}/{gh}/{tif.name}"
            logging.info(f"Uploading {tif.name} → s3://{remote_tif}")
            s3.put(str(tif), remote_tif)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # 8) Upload the metadata CSV to S3
    remote_csv = f"{bucket}/{folder}/{gh}/{metadata_csv.name}"
    logging.info(f"Uploading metadata CSV → s3://{remote_csv}")
    s3.put(str(metadata_csv), remote_csv)
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    # 9) Build climatology layers & band_stats.json
    #    a) Open the newly-written Zarr from S3
    fs = s3fs.S3FileSystem(**storage_options)
    mapper = fs.get_mapper(f"{bucket}/{folder}/{gh}/{gh}_zarr")
    ds = xr.open_zarr(mapper, consolidated=True)

    #    b) Compute and upload seasonal & monthly NDVI climatology
    base_clim = f"s3://{bucket}/{folder}/{gh}/{gh}_climatology"
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
    step += 1
    if progress_callback:
        progress_callback(step / total_steps)

    logging.info(
        f"All done!\n"
        f" • Two dates (8-band GeoTIFFs) are in {out_local}/\n"
        f" • Metadata CSV: {metadata_csv} → uploaded to s3://{remote_csv}\n"
        f" • Zarr on S3: {zarr_path}\n"
        f" • Climatology files (seasonal/monthly NDVI) in: {base_clim}\n"
        f" • Band stats JSON: {stats_path}"
    )

def main():
    """
    Entry point for build_dataset.py:

      1) Parse args & AWS creds
      2) Auth to GEE
      3) For each geohash, check existing Zarr on S3
         a) If exists, update missing dates
         b) Otherwise, download & build full Zarr
      4) Compute and upload climatology & band_stats
    """
    logging.info("Starting build_dataset.py script...")
    args = parse_args()

    # make aws_creds dict
    logging.info("Parsing AWS credentials...")
    if args.aws_creds_file:
        storage_options = make_aws_creds_dict(
            aws_secrets_file=args.aws_creds_file)
    else:
        logging.error("Error: Please provide a valid AWS account key and email")
        sys.exit(1)

    # authenticate google earth engine
    authenticate_ee(account_key=args.ee_account_key, account_email=args.ee_account_email)

    # split geohashes
    geohash_list = [gh.strip() for gh in args.geohashes.split(",")]
    logging.info(f"Requested geohash list: {geohash_list}")

    # for each geohash, check to see if the associated Zarr exists on S3
    existence_list: list[tuple[str, bool, str]] = [ # [ (gh, bool_exists, zarr_path) ... ]
        (gh, *check_existence(
            geohash=gh,
            bucket=args.bucket,
            folder=args.folder,
            storage_options=storage_options)
        )
        for gh in geohash_list
    ]
    
    # only those that do not exist
    existing = [(gh, flag, path) for gh, flag, path in existence_list if flag]
    missing = [(gh, flag, path) for gh, flag, path in existence_list if not flag]
    logging.info(f"Existing zarr list: {existing}")
    logging.info(f"Missing zarr list: {[gh for gh, flag, path in missing if not flag]}")

    # --- existing geohashes --------------------------------------------------
    logging.info("Starting to update existing zarrs on S3...")
    # for each existing geohash, for those that exist, get their date range
    date_ranges = [ # [ (gh, zarr_path, existing_start, existing_end) ... ]
        (gh, path, *get_zarr_date_range(
                        zarr_path=path,
                        storage_options=storage_options,
                        geohash=gh)
        )
        for gh, flag, path in existence_list
        if flag
    ]
    logging.info(f"Date ranges for existing zarrs: {date_ranges}")

    # calculate range still needed to stay current to date_today
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        date_today = end_date
    else:
        date_today = datetime.today().date()
    # Build a list of (geohash, zarr_path, missing_start, missing_end)
    missing_windows = []
    for gh, path, _, end in date_ranges:
        # normalize ANY end (np.datetime64, pd.Timestamp, datetime.datetime or date) → date
        end = pd.to_datetime(end).date()
        if end < date_today:
            missing_start = end + timedelta(days=1)
            missing_windows.append((gh, path, missing_start, date_today))
    logging.info(f"Missing windows list: {missing_windows}")

    for gh, save_path, missing_start, missing_end in missing_windows:
        # # only continue if there really is a gap to fill
        # if end >= date_today:
        #     continue

        logging.info(f"[Update] Updating geohash {gh} from {missing_start} to {missing_end}")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            region = get_ee_region(geohash=gh)

            # 1) pull down the *existing* metadata CSV from S3
            local_meta  = tmpdir / f"metadata_{gh}.csv"
            remote_meta = f"{args.bucket}/{args.folder}/{gh}/metadata_{gh}.csv"
            fs = s3fs.S3FileSystem(**storage_options)
            fs.get(remote_meta, str(local_meta))

            # 2) build an EE collection *just* for the missing window
            coll_new = (
                get_ee_collection(region, cloud_perc=args.cloud_perc)
                .filterDate(
                    (missing_start.strftime("%Y-%m-%d")),
                    (missing_end + timedelta(days=1)).strftime("%Y-%m-%d")
                )
            )

            count = coll_new.aggregate_count('system:index').getInfo()
            logging.info(f"Number of images in new window: {count}")

            # 3) download only those new dates & append into local_meta
            out_dir = tmpdir / gh
            logging.info("[Update] Downloading imagery with metadata")
            # end_date = date_today.strftime("%Y-%m-%d")
            download_all_imagery_with_metadata(
                collection   = coll_new,
                out_dir      = out_dir,
                bbox         = create_bbox(geohash=gh),
                scale        = args.scale,
                region       = region,
                metadata_csv = local_meta,
                geohash      = gh,
                end_date     = ((missing_end + timedelta(days=1)).strftime("%Y-%m-%d")),  # means "up to today"
                skip_static=True
            )

            files = sorted(p.name for p in Path(out_dir).iterdir())
            logging.info(f"Directory listing of {out_dir}:\n" + "\n".join(files))

            # 4) append *just* those new timesteps into the existing Zarr
            logging.info("[Update] Creating zarr from metadata")
            create_zarr_from_imetadata(
                metadata_csv    = local_meta,
                image_dir       = out_dir,
                zarr_path       = save_path,
                storage_options = storage_options,
                append=True,
                start_date=missing_start
            )

            # 5) push the updated metadata_csv back to S3
            fs.put(str(local_meta), remote_meta)
            logging.info(f"Updated metadata CSV uploaded to {remote_meta}")

            fs     = s3fs.S3FileSystem(**storage_options)
            mapper = fs.get_mapper(f"{args.bucket}/{args.folder}/{gh}/{gh}_zarr")
            ds     = xr.open_zarr(mapper, consolidated=True)
            print(f"{gh} zarr times:", ds.time.values)

    # --- missing geohashes ---------------------------------------------------
    # create full datasets for missing geohashes
    logging.info(f"Starting to build full datasets for missing geohashes: {[gh for gh, flag, path in missing]}")
    for gh, _, save_path in missing:
        logging.info(f"Working on geohash: {gh}")
        region = get_ee_region(geohash=gh)
        bbox = create_bbox(geohash=gh)
        coll = get_ee_collection(region=region, cloud_perc=args.cloud_perc)
        with tempfile.TemporaryDirectory() as tmpdir:
            logging.info(f"[DEBUG] tmpdir = {tmpdir}")
            out_dir = Path(tmpdir) / gh
            logging.info(f"[DEBUG] out_dir exists before download? {out_dir.exists()}")

            tmpdir_path = Path(tmpdir)
            out_dir = tmpdir_path / gh
            metadata_csv = tmpdir_path / f"metadata_{gh}.csv"

            logging.info("Downloading imagery with metadata...")
            download_all_imagery_with_metadata(
                collection=coll,
                out_dir=out_dir,
                bbox=bbox,
                scale=args.scale,
                region=region,
                metadata_csv=metadata_csv,
                geohash=gh,
                end_date=args.end_date
            )
            logging.info(f"[DEBUG] out_dir after download: {list(out_dir.iterdir())}")

            # debugging
            slope_path = out_dir / "slope.tif"
            aspect_path = out_dir / "aspect.tif"
            with rio.open(slope_path) as src:
                arr = src.read(1)         # first (and only) band
                print("slope.tif dtype:", arr.dtype)
                print("slope.tif nodata:", src.nodata)
                print("slope.tif min/max (raw):", np.nanmin(arr), np.nanmax(arr))
                # If you want to see edge values:
                h, w = arr.shape
                print("corners:", arr[0,0], arr[0,w-1], arr[h-1,0], arr[h-1,w-1])
            with rio.open(aspect_path) as src:
                arr = src.read(1)         # first (and only) band
                print("aspect.tif dtype:", arr.dtype)
                print("aspect.tif nodata:", src.nodata)
                print("aspect.tif min/max (raw):", np.nanmin(arr), np.nanmax(arr))
                # If you want to see edge values:
                h, w = arr.shape
                print("corners:", arr[0,0], arr[0,w-1], arr[h-1,0], arr[h-1,w-1])

            logging.info(f"Building & writing {gh} Zarr to S3...")
            # save_path should be S3 store path, e.g. "my-bucket/foo/gh.zarr"
            create_zarr_from_imetadata(
                metadata_csv = metadata_csv,
                image_dir = out_dir,
                zarr_path = save_path,
                storage_options = storage_options
            )
            logging.info(f"Done with geohash {gh}; Zarr stored at {save_path}")
            
            # debugging
            # ds = xr.open_zarr("s3:/rgc-zarr-store/data/9vgm5/9vgm5_zarr/", consolidated=True, 
            #                 storage_options=storage_options)
            # sl = ds["slope"].isel(time=0).values
            # print("slope in Zarr dtype:", sl.dtype)
            # print("slope in Zarr min/max:", np.nanmin(sl), np.nanmax(sl))
            # # check a few corner cells:
            # h, w = sl.shape
            # print("corners in Zarr:", sl[0,0], sl[0,w-1], sl[h-1,0], sl[h-1,w-1])

            logging.info(f"Uploading metadata CSV for {gh} to S3...")
            fs = s3fs.S3FileSystem(**storage_options)
            # remote key: bucket/folder/geohash/metadata_geohash.csv
            remote_csv_path = f"{args.bucket}/{args.folder}/{gh}/metadata_{gh}.csv"
            # copy local file up to S3
            fs.put(str(metadata_csv), remote_csv_path)
            logging.info(f"Metadata CSV uploaded to {remote_csv_path}")

            logging.info("Calculating climatology layers for NDVI")
            # open the newly‐written Zarr store
            fs = s3fs.S3FileSystem(**storage_options)
            mapper = fs.get_mapper(f"{args.bucket}/{args.folder}/{gh}/{gh}_zarr")
            ds = xr.open_zarr(mapper, consolidated=True)
            # point at a subfolder for climatology
            clim_dir = f"s3://{args.bucket}/{args.folder}/{gh}/{gh}_climatology"
            # run it
            compute_climatology(
                ds = ds,
                save_dir = clim_dir,
                storage_options= storage_options
            )
            logging.info(f"Climatology layers have been built and uploaded to {clim_dir}")

            # now compute & save band stats alongside the climatology
            stats_path = f"{clim_dir.rstrip('/')}/band_stats.json"
            compute_band_stats(
                ds=ds,
                bands=["R","G","B","NDVI","elevation","slope","aspect", "aspect_mask"],
                stats_path=stats_path,
                storage_options=storage_options
            )
            logging.info(f"Band statistics written to {stats_path}")

            fs     = s3fs.S3FileSystem(**storage_options)
            mapper = fs.get_mapper(f"{args.bucket}/{args.folder}/{gh}/{gh}_zarr")
            ds     = xr.open_zarr(mapper, consolidated=True)
            print(f"{gh} zarr times:", ds.time.values)

    logging.info("No more geohashes to work through. Exiting script.")

if __name__ == "__main__":
    main()

# Example:
# python build_dataset.py \
# 	--bucket rgc-zarr-store \
# 	--folder data \
# 	--ee-account-key secrets/low-res-sat-change-detection-f7e0f971189b.json \
# 	--ee-account-email low-res-sat-change-detection@low-res-sat-change-detection.iam.gserviceaccount.com \
# 	--aws-creds-file secrets/aws_rgc-zarr-store.json \
# 	--geohashes 9vgm0,9vgm1

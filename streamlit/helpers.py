# helpers.py

import math
import streamlit as st
import geohash2
import folium
import os
import s3fs
import torch
import xarray as xr
import pandas as pd
import numpy as np

# variables
variable_list = [
    # 'geohash_text', 
    # 'geohash_bbox', 
    'date0', 
    'date1'
    ]

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points 
    on the Earth specified in decimal degrees of longitude and latitude.
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    km = 6371 * c  # Earth's radius in km
    return km

def reset_button(var_list: list = None):
    """
    Creates a reset button that deletes all session state variables to start fresh
    """
    if var_list is None:
        var_list = variable_list

    with st.sidebar:
        reset = st.button(label="Reset all selections")

    if key in var_list:
        for key in var_list:
            del st.session_state[key]
        st.rerun()

    return reset
    
def states_and_regions():
    return [
    # 50 U.S. states
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",

    # Additional regions / territories
    "District of Columbia",     # Washington, D.C.
    "Puerto Rico",
    "Guam",
    "American Samoa",
    "U.S. Virgin Islands",
    "Northern Mariana Islands"
]

def state_selector():
    """
    Creates a dropdown/text input selector 
    to filter the desired geohashes by state
    """
    state_selection = st.selectbox(
        label="Select state or region",
        options=states_and_regions(),
        index=0
    )

    return state_selection

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

def geohash_to_centroid(gh: list | str, show_map: bool = True, m: folium.Map = None):
    """
    Returns centroid(s) of geohash(es) with an optional Folium map.
    If an existing map 'm' is provided, markers are added to it. Otherwise, a new map is created.

    Args:
        gh (list or str): A list of geohash strings or a single geohash string.
        show_map (bool): If True, markers are added to a Folium map.
        m (folium.Map, optional): An existing Folium map to add markers to. Defaults to None.

    Returns:
        If show_map is True, returns a tuple (centroids, m) where:
          - centroids is a list of (lat, lon) tuples.
          - m is the Folium Map object with added markers.
        Otherwise, returns just the list of (lat, lon) tuples.
    """
    # Ensure gh is a list of strings
    if isinstance(gh, str):
        geohash_list = [gh]
    elif isinstance(gh, list):
        if all(isinstance(item, str) for item in gh):
            geohash_list = gh
        else:
            raise ValueError("All items in the list must be strings.")
    else:
        raise ValueError("gh must be either a string or a list of strings.")
    
    # Decode each geohash to get (lat, lon, error1, error2)
    decoded = [geohash2.decode_exactly(gh_val) for gh_val in geohash_list]
    
    # Unpack only the latitude and longitude values
    centroids = [(lat, lon) for (lat, lon, _, _) in decoded]
    
    if show_map:
        if m is None:
            # Create a new map centered around the average coordinates
            avg_lat = sum(lat for lat, _ in centroids) / len(centroids)
            avg_lon = sum(lon for _, lon in centroids) / len(centroids)
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)
        # Add a marker for each centroid
        for lat, lon in centroids:
            folium.Marker(location=[lat, lon]).add_to(m)
        return centroids, m
    
    return centroids

def bbox_centroid(bbox: dict) -> dict:
    """
    Given a bbox dict with keys 'xmin','ymin','xmax','ymax',
    returns a dict with the centroid’s latitude and longitude.
    """
    lon_center = (bbox['xmin'] + bbox['xmax']) / 2
    lat_center = (bbox['ymin'] + bbox['ymax']) / 2
    # return {'lat': lat_center, 'lon': lon_center}
    return [lat_center, lon_center]

def load_slide_paths(folder: str) -> list[str]:
    """Return a sorted list of slide file paths (PNG/JPG) in the folder."""
    files = os.listdir(folder)
    # keep only .png/.jpg/.jpeg, sorted by filename
    imgs = sorted([f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    return [os.path.join(folder, img) for img in imgs]

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

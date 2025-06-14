# 1_Presentation_and_Demo.py

import os
import streamlit as st
from datetime import date, timedelta, datetime
from streamlit_folium import st_folium
import folium
from streamlit.components.v1 import html
from pathlib import Path
import json
import s3fs
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

import helpers as h
from build_dataset_streamlit import run_for_geohash
from inspect_dataset_streamlit import inspect_dataset_for_streamlit
from inference_streamlit import run_inference_for_geohash

def main():
    st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
    st.header("Low-Resolution Satellite Image Change Detection Model | Samuel Alter | Reveal Global Consulting")

    # ─── Create two equal columns ─────────────────────────────────────────────
    col_left, col_right = st.columns([0.6, 0.4], gap="small")
    MAP_HEIGHT = 350

    # ─── Left column: embedded Google Slides ──────────────────────────────────
    with col_left:
        st.subheader("Presentation")
        GOOGLE_SLIDES_EMBED_SRC = (
            "https://docs.google.com/presentation/d/e/"
            "2PACX-1vSoyXNvBjV7c5gyXdsRQPOvn_irAoEBtAaFYOiGk1EthHOYoLoWUr70hsPEe-"
            "V9tNeo3jkDcPDVgRg9/pubembed?start=true&loop=false&delayms=60000"
        )
        iframe_code = f"""
            <iframe
                src="{GOOGLE_SLIDES_EMBED_SRC}"
                frameborder="0"
                width="100%"
                height="600px"
                allowfullscreen="true"
                mozallowfullscreen="true"
                webkitallowfullscreen="true">
            </iframe>
        """
        html(iframe_code, height=600)

    # ─── Right column: interactive demo ───────────────────────────────────────
    with col_right:
        st.subheader("Demo")

        # 1) Define some constants & “today” for bounds
        CENTER_START = [40, -100]
        ZOOM_START = 4
        TODAY = datetime.today().date()

        # 2) Initialize session_state keys if they don’t exist yet
        if "center" not in st.session_state:
            st.session_state["center"] = CENTER_START
        if "zoom" not in st.session_state:
            st.session_state["zoom"] = ZOOM_START
        if "in_gh" not in st.session_state:
            st.session_state["in_gh"] = ""           # holds the geohash
        if "date0" not in st.session_state:
            st.session_state["date0"] = TODAY - timedelta(days=1)
        if "date1" not in st.session_state:
            st.session_state["date1"] = TODAY
        if "built" not in st.session_state:
            st.session_state["built"] = False        # toggles after build finishes
        if "inspected" not in st.session_state:
            st.session_state["inspected"] = False    # toggles after inspect finishes
        if "inferred" not in st.session_state:
            st.session_state["inferred"] = False     # toggles after inference finishes

        # ─── “Reset” button at top of right column ────────────────────────────
        c1_reset, c2_reset = st.columns([3, 1], gap="small")
        with c2_reset:
            # push the Reset button down a bit
            st.markdown("<div style='margin-top: 28px'></div>", unsafe_allow_html=True)
            if st.button("Reset"):
                # clear everything
                for key in [
                    "in_gh",
                    "date0",
                    "date1",
                    "center",
                    "zoom",
                    "built",
                    "inspected",
                    "inferred",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                # restore defaults
                st.session_state["center"] = CENTER_START
                st.session_state["zoom"] = ZOOM_START
                st.session_state["date0"] = TODAY - timedelta(days=1)
                st.session_state["date1"] = TODAY
                st.session_state["built"] = False
                st.session_state["inspected"] = False
                st.session_state["inferred"] = False
                st.rerun()

        # ─── Geohash text_input ──────────────────────────────────────────────
        with c1_reset:
            in_gh = st.text_input(
                label="Geohash (5 characters)",
                max_chars=5,
                key="in_gh",
                placeholder="e.g. 9vgm0. Go to https://www.geohash.es/browse/ to explore.",
            )

        # ─── If the user typed a 5-char value, validate & recenter the map ───
        bbox = None
        if in_gh and len(in_gh) == 5:
            try:
                _ = h.geohash2.decode_exactly(in_gh)
                bbox = h.create_bbox(in_gh)
                c = h.bbox_centroid(bbox)
                st.session_state["center"] = [c[0], c[1]]
                st.session_state["zoom"] = 11
            except KeyError:
                st.error(f"“{in_gh}” is not a valid 5-character geohash.")

        # ─── Always draw the Folium map here ──────────────────────────────────
        m = folium.Map(
            location=st.session_state["center"],
            zoom_start=st.session_state["zoom"],
        )
        # add Esri satellite & labels
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Esri Satellite",
            overlay=True,
            control=False,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Boundaries and Places",
            name="Esri Labels",
            overlay=True,
            control=False,
        ).add_to(m)
        if bbox:
            fg = folium.FeatureGroup(name="GeohashBBox")
            sw = [bbox["ymin"], bbox["xmin"]]
            ne = [bbox["ymax"], bbox["xmax"]]
            rect = folium.Rectangle(bounds=[sw, ne], color="blue", weight=4, fill=False)
            fg.add_child(rect)
            st_folium(
                m,
                center=st.session_state["center"],
                zoom=st.session_state["zoom"],
                feature_group_to_add=fg,
                height=MAP_HEIGHT,
                use_container_width=True,
                key="map1",
            )
        else:
            st_folium(
                m,
                center=st.session_state["center"],
                zoom=st.session_state["zoom"],
                height=MAP_HEIGHT,
                use_container_width=True,
                key="map1",
            )

        # ─── STEP 2: Show date pickers + Build button *only* if a geohash is entered ──────
        if in_gh:
            col_d0, col_d1 = st.columns(2)
            with col_d0:
                # “Before” date
                date0 = st.date_input(
                    "Select first date (before)",
                    key="date0",
                    min_value=date(2018, 1, 1),
                    max_value=TODAY - timedelta(days=1),
                )
            with col_d1:
                # “After” date
                date1 = st.date_input(
                    "Select second date (after)",
                    key="date1",
                    min_value=date(2019, 1, 1),
                    max_value=TODAY,
                )

            if date1 <= date0:
                st.error("Second date must be strictly after the first date.")
            else:
                # only enable “Build dataset” if the dates are valid
                build_button = st.button(
                    f"**Build** data for **{in_gh}** on **{date0}** → **{date1}**"
                )
                if build_button:
                    # prevent double‐click from re‐setting built prematurely
                    st.session_state["built"] = False
                    st.session_state["inspected"] = False
                    st.session_state["inferred"] = False

                    with st.spinner("Running build pipeline (upload → S3)…"):
                        progress_bar = st.progress(0)
                        try:
                            _ = run_for_geohash(
                                geohash=in_gh,
                                date0=date0.strftime("%Y-%m-%d"),
                                date1=date1.strftime("%Y-%m-%d"),
                                progress_callback=lambda frac: progress_bar.progress(frac),
                            )
                            st.success("Build pipeline completed and pushed to S3.")
                            st.session_state["built"] = True
                        except Exception as e:
                            st.error(f"Build pipeline failed: {e}")
                            return

            # ─── STEP 3: Show “Inspect dataset” *only* after build is done ─────────────
            if st.session_state.get("built", False):
                st.info("Building the dataset has finished. Now you can inspect that dataset.")
                inspect_button = st.button(f"Inspect dataset for **{in_gh}** on **{date0}** → **{date1}**")
                if inspect_button:
                    st.session_state["inspected"] = False
                    st.session_state["inferred"] = False

                    with st.spinner("Running inspect pipeline…"):
                        progress_bar2 = st.progress(0)
                        try:
                            ins_out = inspect_dataset_for_streamlit(
                                geohash=in_gh,
                                date0_str=date0.strftime("%Y-%m-%d"),
                                date1_str=date1.strftime("%Y-%m-%d"),
                                progress_callback=lambda frac: progress_bar2.progress(frac),
                            )
                            st.success("Inspect pipeline completed.")
                            st.session_state["inspected"] = True
                            st.session_state["ins_out"] = ins_out
                        except Exception as e:
                            st.error(f"Inspect pipeline failed: {e}")
                            return

            # ─── STEP 4: Show “Run inference” *only* after inspect is done ───────────────
            if st.session_state.get("inspected", False):
                st.info("Inspect finished. Now you can run inference on that dataset.")
                infer_button = st.button(f"Run inference for **{in_gh}** on **{date0}** → **{date1}**")
                if infer_button:
                    st.session_state["inferred"] = False

                    with st.spinner("Running inference…"):
                        progress_bar3 = st.progress(0)
                        try:
                            inf_out = run_inference_for_geohash(
                                geohash=in_gh,
                                date0_str=date0.strftime("%Y-%m-%d"),
                                date1_str=date1.strftime("%Y-%m-%d"),
                                progress_callback=lambda frac: progress_bar3.progress(frac),
                            )
                            st.success("Inference is complete.")
                            st.session_state["inferred"] = True
                            st.session_state['inf_out'] = inf_out
                        except Exception as e:
                            st.error(f"Inference failed: {e}")
                            return

            # ─── STEP 5: Once “inferred” is True, display all outputs ───────────────────
            if st.session_state.get("inferred", False):
                inf_out = st.session_state.get("inf_out", None)
                if inf_out is None:
                    st.error("Something went wrong: no inference outputs found.")
                else:
                    st.info("Inference has finished. Scroll down below to view all inspection and inference assets.")
                    st.markdown("---")
                    st.markdown("## Inspection & Inference Results")

                    # ─── First: Inspection section ────────────────────────────────
                    st.markdown("### Inspection PNGs")
                    base_inspect_folder = Path("inspect_outputs") / f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}"
                    rgb0      = base_inspect_folder / "rgb_t0.png"
                    rgb1      = base_inspect_folder / "rgb_t1.png"
                    ndvi_diff = base_inspect_folder / "ndvi_diff.png"

                    # Two-column for RGB t₀ / t₁
                    insp_rgb = st.columns(2)
                    with insp_rgb[0]:
                        if rgb0.exists():
                            st.image(str(rgb0), caption="RGB at t₀", use_container_width=True)
                    with insp_rgb[1]:
                        if rgb1.exists():
                            st.image(str(rgb1), caption="RGB at t₁", use_container_width=True)

                    # If the precomputed ndvi_diff.png exists, show it. Otherwise, recompute in-memory and plot t₀, t₁, Δ side-by-side.
                    if ndvi_diff.exists():
                        st.image(str(ndvi_diff), caption="NDVI(t₁) − NDVI(t₀)", use_container_width=True)
                    else:
                        # Load the Zarr again (just to pull NDVI)
                        creds = None
                        if os.path.exists("secrets/aws_rgc-zarr-store.json"):
                            creds = json.load(open("secrets/aws_rgc-zarr-store.json", "r"))
                        s3_path = f"rgc-zarr-store/data/{in_gh}/{in_gh}_zarr_streamlit"
                        opts = {}
                        if creds:
                            opts = {"key": creds["access_key"], "secret": creds["secret_access_key"]}
                        fs = s3fs.S3FileSystem(anon=not bool(creds), **opts)
                        mapper = fs.get_mapper(s3_path)
                        ds = xr.open_zarr(mapper, consolidated=True)

                        time_dim = next(d for d in ds.dims if "time" in d.lower() or "date" in d.lower())
                        lat_dim  = next(d for d in ds.dims if "lat" in d.lower() or d.lower() == "y")
                        lon_dim  = next(d for d in ds.dims if "lon" in d.lower() or d.lower() == "x")

                        # Recompute nearest indices for date0/date1
                        def _to_idx(date_str: str):
                            arr = ds[time_dim].values.astype("datetime64[D]")
                            try:
                                tgt = np.datetime64(pd.to_datetime(date_str).date(), "D")
                            except:
                                if date_str.lower() == "first": return 0
                                if date_str.lower() == "last":  return len(arr) - 1
                                tgt = np.datetime64(date_str, "D")
                            deltas = np.abs(arr - tgt)
                            idx = int(deltas.argmin())
                            idx = max(0, min(idx, len(arr)-1))
                            return idx

                        t0_idx = _to_idx(date0.strftime("%Y-%m-%d"))
                        t1_idx = _to_idx(date1.strftime("%Y-%m-%d"))
                        if t1_idx <= t0_idx:
                            t1_idx = min(t0_idx+1, len(ds[time_dim]) - 1)

                        # Pull out NDVI arrays
                        full0 = ds["NDVI"].isel({time_dim: t0_idx}).values
                        full1 = ds["NDVI"].isel({time_dim: t1_idx}).values
                        diff_ndvi = full1 - full0

                        # Build 1×3 figure: [NDVI t₀ | NDVI t₁ | ΔNDVI]
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        vmin, vmax = -1, 1
                        titles = [f"NDVI t₀ ({date0.strftime("%Y-%m-%d")})", f"NDVI t₁ ({date1.strftime("%Y-%m-%d")})", "ΔNDVI"]
                        arrays = [full0, full1, diff_ndvi]
                        cmap  = "RdYlGn"
                        for ax, arr, title in zip(axes, arrays, titles):
                            im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
                            ax.set_title(title)
                            ax.axis("off")
                            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

                        st.pyplot(fig)
                        plt.close(fig)

                    # ─── Add elevation/slope/aspect section ──────────────────────────
                    st.markdown("### Static‐Band Maps (Elevation, Slope, Aspect)")
                    # If we already saved static‐band PNGs in the CLI version, you could display those directly.
                    # Otherwise, re-load and plot here:

                    # (Re-open the same Zarr to pull those static bands:)
                    creds = None
                    if os.path.exists("secrets/aws_rgc-zarr-store.json"):
                        creds = json.load(open("secrets/aws_rgc-zarr-store.json", "r"))
                    s3_path = f"rgc-zarr-store/data/{in_gh}/{in_gh}_zarr_streamlit"
                    opts = {}
                    if creds:
                        opts = {"key": creds["access_key"], "secret": creds["secret_access_key"]}
                    fs = s3fs.S3FileSystem(anon=not bool(creds), **opts)
                    mapper = fs.get_mapper(s3_path)
                    ds = xr.open_zarr(mapper, consolidated=True)

                    # Elevation, slope, aspect might be 2D or 3D (time, y, x). We’ll take the first time‐slice if needed:
                    def _get_static(varname: str):
                        da = ds[varname]
                        if "time" in da.dims:
                            arr = da.isel({time_dim: 0}).values
                        else:
                            arr = da.values
                        return arr

                    elev  = _get_static("elevation")
                    slope = _get_static("slope")
                    aspect = _get_static("aspect")

                    # Build 1×3 figure for elevation, slope, aspect:
                    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
                    static_vars = [("Elevation (m)", elev, "terrain"),
                                ("Slope (°)",    slope, "viridis"),
                                ("Aspect (°)",   aspect, "viridis")]
                    # Determine spatial extents so axis labels are meaningful:
                    lats = ds[lat_dim].values
                    lons = ds[lon_dim].values
                    full_extent = [
                        float(lons.min()), float(lons.max()),
                        float(lats.min()), float(lats.max())
                    ]

                    for ax, (title, arr, cmap) in zip(axes2, static_vars):
                        im = ax.imshow(arr,
                                    extent=full_extent,
                                    origin="upper",
                                    cmap=cmap)
                        ax.set_title(title)
                        ax.set_xlabel("Longitude")
                        ax.set_ylabel("Latitude")
                        ax.grid(True, linestyle="--", alpha=0.4)
                        fig2.colorbar(im, ax=ax, shrink=0.7, label=title)

                    st.pyplot(fig2)
                    plt.close(fig2)

                    # ─── Inspection Summary Tables ─────────────────────────────────
                    st.markdown("### Inspection Summary Tables")
                    time_dim = next(d for d in ds.dims if "time" in d.lower() or "date" in d.lower())

                    # 1) time_summary → DataFrame
                    ts = h.compute_time_summary(ds[time_dim]).to_frame().T
                    st.write("**Time dimension summary:**")
                    st.dataframe(ts)

                    # 2) band_summary → DataFrame
                    bs = h.compute_band_summary(ds).round(3)
                    st.write("**Per‐band summary stats:**")
                    st.dataframe(bs)

                    # ─── Download buttons for Inspection assets ─────────────────────────
                    st.markdown("#### Download Inspection Assets")

                    # (a) RGB t₀
                    if rgb0.exists():
                        with open(rgb0, "rb") as f:
                            bytes_rgb0 = f.read()
                        st.download_button(
                            label="Download rgb_t0.png",
                            data=bytes_rgb0,
                            file_name=f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}_rgb_t0.png",
                            mime="image/png",
                        )

                    # (b) RGB t₁
                    if rgb1.exists():
                        with open(rgb1, "rb") as f:
                            bytes_rgb1 = f.read()
                        st.download_button(
                            label="Download rgb_t1.png",
                            data=bytes_rgb1,
                            file_name=f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}_rgb_t1.png",
                            mime="image/png",
                        )

                    # (c) NDVI diff
                    if ndvi_diff.exists():
                        # If we already saved ndvi_diff.png to disk, read it and let the user download it.
                        with open(ndvi_diff, "rb") as f:
                            bytes_ndvi = f.read()
                        st.download_button(
                            label="Download ndvi_diff.png",
                            data=bytes_ndvi,
                            file_name=f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}_ndvi_diff.png",
                            mime="image/png",
                        )
                    else:
                        # We generated a “fig” in the NDVI‐recompute section above. Capture that plot into a BytesIO
                        buf = io.BytesIO()
                        # (Important: the variable “fig” is the figure object you built for [NDVI t₀ | NDVI t₁ | ΔNDVI].)
                        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
                        buf.seek(0)
                        bytes_ndvi = buf.getvalue()
                        st.download_button(
                            label="Download ndvi_diff.png",
                            data=bytes_ndvi,
                            file_name=f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}_ndvi_diff.png",
                            mime="image/png",
                        )
                        buf.close()

                    # (d) Static-band images (save to disk if you want to persist; otherwise, skip)
                    # Let’s write the static‐band figure we just built to disk so it can be downloaded:
                    static_png = base_inspect_folder / "static_bands.png"
                    fig2.savefig(static_png, bbox_inches="tight", pad_inches=0.1)
                    if static_png.exists():
                        with open(static_png, "rb") as f_sb:
                            bytes_sb = f_sb.read()
                        st.download_button(
                            label="Download static_bands.png",
                            data=bytes_sb,
                            file_name=f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}_static_bands.png",
                            mime="image/png",
                        )

                    # (e) time_summary.csv
                    time_csv_path = base_inspect_folder / "time_summary.csv"
                    ts.to_csv(time_csv_path)
                    with open(time_csv_path, "rb") as f_ts:
                        bytes_ts = f_ts.read()
                    st.download_button(
                        label="Download time_summary.csv",
                        data=bytes_ts,
                        file_name=f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}_time_summary.csv",
                        mime="text/csv",
                    )

                    # (f) band_summary.csv
                    band_csv_path = base_inspect_folder / "band_summary.csv"
                    bs.to_csv(band_csv_path)
                    with open(band_csv_path, "rb") as f_bs:
                        bytes_bs = f_bs.read()
                    st.download_button(
                        label="Download band_summary.csv",
                        data=bytes_bs,
                        file_name=f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}_band_summary.csv",
                        mime="text/csv",
                    )

                    # ─── Now: Inference section ───────────────────────────────
                    st.markdown("---")
                    st.markdown("### Inference PNGs & Overlays")
                    inf_folder = Path("inference_outputs") / f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}"

                    cols_inf = st.columns(2)
                    with cols_inf[0]:
                        st.image(inf_out["rgb_t0"],            caption="RGB t₀",            use_container_width=True)
                        st.image(inf_out["heatmap"],           caption="Δ-heatmap",        use_container_width=True)
                        st.image(inf_out["overlay_heatmap_t0"], caption="Overlay: Δμ on t₀",    use_container_width=True)

                    with cols_inf[1]:
                        st.image(inf_out["rgb_t1"],            caption="RGB t₁",            use_container_width=True)
                        st.image(inf_out["mask"],              caption="Change Mask",       use_container_width=True)
                        st.image(inf_out["overlay_mask_t1"],   caption="Overlay: Mask on t₁",  use_container_width=True)

                    st.markdown("#### Download Inference Assets")
                    # (a) PDF report
                    with open(inf_out["report_pdf"], "rb") as f_pdf:
                        pdf_bytes = f_pdf.read()
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}_report.pdf",
                        mime="application/pdf",
                    )
                    # (b) summary_statistics.csv
                    with open(inf_out["stats_csv"], "rb") as f_csv:
                        csv_bytes = f_csv.read()
                    st.download_button(
                        label="Download summary_statistics.csv",
                        data=csv_bytes,
                        file_name=f"{in_gh}_{date0.strftime('%Y%m%d')}_{date1.strftime('%Y%m%d')}_summary_statistics.csv",
                        mime="text/csv",
                    )
        else:
            # If the geohash box is empty, prompt the user to enter one.
            st.info("Enter a 5-character Geohash above to see date pickers & Run buttons.")


if __name__ == "__main__":
    main()

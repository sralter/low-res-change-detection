# low-res-change-detection

By Samuel Alter

The project was completed with the support of [Reveal Global Consulting](https://www.revealgc.com).  
Thank you to the whole team for their helpful comments and contributions.

## Overview

This repository provides a proof-of-concept pipeline for training a change detection model using low-resolution satellite imagery. The key tools used include Google Earth Engine, xarray, and PyTorch. The workflow consists of data collection, preprocessing, model training (including hyperparameter search), inference, and optional visualization of the neural network architecture.

The model integrates the following data:
* True color imagery
* NDVI
* Elevation
* Slope
* Aspect

## Table of Contents <a name='table-of-contents'></a>

1. [Requirements](#requirements)
   * [Dependencies](#dependencies)
2. [Project Structure](#project_structure)
3. [Secrets Configuration](#secrets)
4. [Usage](#usage)
   * [Containerized Usage with Docker](#container)
   * [Running the Scripts](#script)
      * [Build Dataset](#build_dataset)
      * [Inspect Dataset](#inspect_dataset)
      * [Train Model](#train_vae)
      * [Run Hyperparameter Search](#hp_search)
      * [Inference](#inference)
      * [Visualize Neural Network (optional)](#visualize_nn)
5. [For Further Reading](#further)

## 1. Requirements <a name='requirements'></a>

[Back to TOC](#table-of-contents)

* Python 3.12.9+
* Google Earth Engine Python API Key (see [below](#gee))
* AWS credentials JSON (see [below](#gee))
* xarray, rasterio, rio-xarray
* PyTorch
* Optuna (for hyperparameter search)
* numpy, pandas, matplotlib
* streamlit (optional, for front-end interaction)

### Install

```bash
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

### API Keys <a name='gee'></a>

[Back to TOC](#table-of-contents)

**For Google Earth Engine**:  
Go to the [Earth Engine Apps guide](https://developers.google.com/earth-engine/guides/app_key) and create an API key.

**For AWS**:
We recommend creating your own S3 bucket when saving datasets to the cloud. Follow AWS' instructions on how to retrieve the bucket's credentials for your use.

### Install dependencies <a name='dependencies'></a>

[Back to TOC](#table-of-contents)

Use the provided requirements.txt to install the dependencies for this project:

```bash
uv pip install -r requirements.txt
```

## 2. Project Structure <a name='project_structure'></a>

[Back to TOC](#table-of-contents)

```text
├── documents/                  # Project reports, presentations, and other literature
└── results/                    # Where the best trained model and older training runs are saved
    ├── _old/                   # Older training runs are saved here
    └── hp_1/                   # Best trained model, inference, and tensorboard files
└── secrets_templates/          # Template secret files for credentials
    ├── aws_creds_file_template.json
    ├── google_earth_engine_creds_file_template.json
    └── email_creds_file_template.json
└── sources/                    # Folders for other projects that inspired this one
└── streamlit/                  # Files supporting the presentation and project demo
    ├── Home.py                 # Script that defines Streamlit dashboard
    ├── best_model.pt           # Copy of best model found in hp_1 folder above
    ├── build_dataset_streamlit.py      # Streamlit-specific script of build_dataset.py
    ├── helpers.py              # Script with helper functions for Streamlit app
    ├── inference_streamlit.py  # Streamlit-specific script of inference.py
    ├── inspect_dataset_streamlit.py    # Streamlit-specific script of inspect_dataset.py
    ├── prepare_dataset.py      # Copy of prepare_dataset.py for Streamlit app
    ├── vae.py                  # Copy of vae.py for Streamlit app
    └── pages/
        ├── 1_Presentation_and_Demo.py  # Script to show Google Doc presentation on one side and demo on the other
        └── 2_Only_Presentation.py      # Script to show just the Google Doc presentation
├── .gitignore.txt              # .gitignore document
├── README.md                   # README document
├── build_dataset.py            # Collect and organize Sentinel-2 imagery via GEE and xarray into a dataset
├── hp_search.py                # Entry point: hyperparameter search orchestration with Optuna
├── inference.py                # Run inference on new imagery using trained model
├── inspect_dataset.py          # Inspect and summarize dataset statistics
├── prepare_dataset.py          # Further dataset preparation (e.g., NDVI computation)
├── requirements.txt            # Python dependencies
├── train_vae.py                # Entry point: train VAE model in one training run
├── trainer.py                  # Trainer utilities (logging, checkpointing, metrics)
├── vae.py                      # Define the VAE model architecture
├── visualize_nn.py             # (Optional) Visualize network architecture
```

## 3. Secrets Configuration <a name='secrets'></a>

[Back to TOC](#table-of-contents)

You will need to get an AWS credentials file, a Google Earth Engine account, and if you want the `hp_search.py` script to send you an email, you will also need to set up an app-specific password for your mail client.

## 4. Usage <a name='usage'></a>

[Back to TOC](#table-of-contents)

### Containerized Usage with Docker <a name='container'></a>

[Back to TOC](#table-of-contents)

We’ve provided a `Dockerfile` and `docker-compose.yml` so you can “clone → build → run” without worrying about system setup.

The workflow is simple:
1. Build the docker image
   ```bash
   git clone https://github.com/sralter/low-res-change-detection.git
   cd low-res-change-detection
   docker compose up --build -d
   ```
2. Execute the scripts
   ```bash
   docker compose run --rm app build_dataset.py [your args …]
   ```

### Running the Scripts <a name='script'></a>

[Back to TOC](#table-of-contents)

Refer to the below sections when running each process. All scripts are run from the command line. Our guidance is to run the following scripts in this order. With `train_vae` and `hp_search`, you can choose either a single training run with `train_vae.py`, or a hyperparameter search with `hp_search.py`.
1. [Build Dataset.py](#build_dataset)
2. [Inspect Dataset.py](#inspect_dataset)
3. [Train VAE](#train_vae) **OR** [HP Search](#hp_search)
4. [Inference](#inference)
5. [Visualize NN](#visualize_nn)

#### 1. Build Dataset <a name='build_dataset'></a>

[Back to TOC](#table-of-contents)

```bash
python build_dataset.py \
    --bucket rgc-zarr-store \
    --folder data \
    --ee-account-key secrets/low-res-sat-change-detection-f7e0f971189b.json \
    --ee-account-email low-res-sat-change-detection@low-res-sat-change-detection.iam.gserviceaccount.com \
    --aws-creds-file secrets/aws_rgc-zarr-store.json \
    --geohashes 9vgm0,9vgm1
```

#### 2. Inspect Dataset <a name='inspect_dataset'></a>

[Back to TOC](#table-of-contents)

```bash
python inspect_dataset.py \
    --bucket rgc-zarr-store \
    --folder data \
    --geohashes 9v1z2 \
    --output outputs \
    --aws-creds secrets/aws_rgc-zarr-store.json \
    --dates first last
```

#### 3a. Train Model <a name='train_vae'></a>

[Back to TOC](#table-of-contents)

We recommend using the `hp_search.py` script first so that Optuna can search the hyperparameter space, then run those best parameters in `train_vae.py`

```bash
python train_vae.py \
    --bucket rgc-zarr-store \
    --folder data \
    --train-geohashes 9vgm0,9vgm1,9vgm2 \
    --val-geohashes 9vgm3 \
    --test-geohashes 9vgm4 \
    --out results \
    --aws-creds secrets/aws_rgc-zarr-store.json \
    --num-workers 4 \
    --stage-zarr
```

#### 3b. Run Hyperparameter Search <a name='hp_search'></a>

[Back to TOC](#table-of-contents)

```bash
python hp_search.py \
    --trials 20 \
    --bucket rgc-zarr-store \
    --folder data \
    --train-geohashes 9vgm0,9vgm1,9vgm2 \
    --val-geohashes 9vgm3 \
    --test-geohashes 9vgm4 \
    --out results/hp_run_1 \
    --aws-creds secrets/aws_rgc-zarr-store.json \
    --num-workers 0 \
    --stage-zarr \
    --trial-epochs 5 \
    --trial-patience 3
```

#### 4. Inference <a name='inference'></a>

[Back to TOC](#table-of-contents)

```bash
python inference.py \
    --bucket rgc-zarr-store \
    --folder data \
    --geohash 9vgm0 \
    --model path/to/best/model.pt \
    --dates first last \
    --aws-creds-file secrets/aws_rgc-zarr-store.json \
    --out outputs \
    --stage-zarr
```

#### 5. Visualize Neural Network (Optional) <a name='visualize_nn'</a>

[Back to TOC](#table-of-contents)

```bash
python visualize_nn.py \
    --model path/to/best/model.pt \
```

## 5. Further Reading <a name='further'></a>

[Back to TOC](#table-of-contents)

If you want to learn more about the project and learn the "how" and "why", I urge you to read the final project report and look at the slideshow presentation, both in the documents folder.
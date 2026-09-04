# Stain Normalisation Does Not Consistently Improve Matched-Location Retrieval with Pathology Foundation Models

Code for the paper submitted to the 7th International Conference on Medical Imaging and Computer-Aided Diagnosis (MICAD 2026), Springer LNEE.

**Authors:** Yaning Meng, Binghao Chai, Tapabrata Chakraborti

## Overview

This repository contains code for evaluating whether stain normalisation improves tile-level image retrieval with DINOv2-based pathology foundation models (Phikon-v2 and Midnight-12k) using the PLISM dataset.

## Repository Structure

```
stain-normalisation-retrieval/
├── README.md
├── environment.yml                 # Conda environment specification
├── setup_env.sh                    # Create/update and verify the Conda environment
├── plism_index.parquet             # Pre-built shard index for PLISM
├── plism_slides.csv                # Slide metadata (stainer, scanner)
├── scripts/
│   ├── plism_loader.py             # Download and tile PLISM WSIs
│   ├── extract_features.py         # Stain normalisation + feature extraction
│   ├── evaluate_retrieval.py       # Paired top-1 nearest-neighbour retrieval
│   ├── plot_retrieval.py           # Figure 2: retrieval accuracy bar charts (with 95% CI)
│   └── plot_umap.py               # Figure 3: UMAP of slide-level embeddings
└── notebooks/
    ├── phikon_v2_pipeline.ipynb    # Phikon-v2: tiling + extraction + evaluation
    └── midnight_12k_pipeline.ipynb # Midnight-12k: extraction + retrieval + UMAP
```

## Setup

```bash
bash setup_env.sh
conda activate plism
```

Alternatively, create the environment directly with
`conda env create -f environment.yml`.

## Usage

### 1. Download and Tile PLISM

The repository includes the pre-built shard index and slide manifest, so the
dataset subset can be fetched directly:

```bash
python scripts/plism_loader.py fetch \
    --tiles-per-slide 400 \
    --out-dir tiles
```

This writes one HDF5 file per slide. Each file contains the RGB tile array,
the corresponding PLISM tile IDs, and the slide/stainer/scanner metadata. The
same tile IDs are selected for every slide, preserving the matched-location
comparisons used by the retrieval analysis.

### 2. Extract Features

```bash
# No normalisation
python scripts/extract_features.py \
    --tiles-dir tiles --model phikon_v2 \
    --normalise none --out-dir features/phikon_v2_none

# Reinhard
python scripts/extract_features.py \
    --tiles-dir tiles --model phikon_v2 \
    --normalise reinhard --out-dir features/phikon_v2_reinhard

# Macenko
python scripts/extract_features.py \
    --tiles-dir tiles --model phikon_v2 \
    --normalise macenko --out-dir features/phikon_v2_macenko
```

Replace `phikon_v2` with `midnight` for Midnight-12k.

### 3. Retrieval Evaluation

```bash
python scripts/evaluate_retrieval.py \
    --feature-dir features/phikon_v2_none \
    --run-name phikon_v2_none \
    --out-dir results
```

Retrieval is computed in both directions for each slide pair and averaged.

### 4. Figure Generation

```bash
python scripts/plot_retrieval.py --out figures/fig2.pdf
python scripts/plot_umap.py --feature-dir features/midnight_none --out figures/fig3.pdf
```

## Data

The PLISM dataset is publicly available: [Ochi et al., Scientific Data, 2024](https://doi.org/10.1038/s41597-024-03122-5).
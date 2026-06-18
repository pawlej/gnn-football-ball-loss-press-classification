# GNN Football Ball Loss & Pressing Classification

A Graph Neural Network (GNN) approach to classifying **ball loss** and **pressing events** in football, based on player positions on the pitch represented as graph structures.

---

## Overview

This project leverages the spatial relationships between players on the pitch to classify two key tactical events:

- **Ball loss** — moments when a team loses possession of the ball
- **Pressing** — coordinated high-pressure defending by a team

Player positions at each event frame are modeled as a graph, where nodes represent players and edges encode proximity or spatial relationships. A GNN is then trained to classify these tactical events.

**Pipeline:**
```
Raw event data → Data preprocessing → Graph construction → GNN training & evaluation
```

---

## Project Structure

```
├── data_preprocessing.ipynb       # Load and clean raw event/tracking data
├── building_graphs_to_train.ipynb # Convert player positions into graph structures
├── model_training.ipynb           # Define, train, and evaluate the GNN model
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/pawlej/gnn-football-ball-loss-press-classification.git
cd gnn-football-ball-loss-press-classification
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch Geometric

PyG requires a separate installation step matching your PyTorch and CUDA version. Check your versions first:

```bash
python -c "import torch; print(torch.__version__)"
```

Then install PyG (example for CPU / CUDA 11.8 — adjust accordingly):

```bash
# CPU only
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html

# or with CUDA (e.g. cu118)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.x.x+cu118.html
```

> See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

### 5. Launch Jupyter

```bash
jupyter notebook
```

---

## Data

This project uses **StatsBomb open data** (or compatible event/tracking data). Data files (`.json`, `.csv`, `.parquet`) are excluded from the repository via `.gitignore`.

To get the data:

```bash
pip install statsbombpy
```

Or download directly from [StatsBomb Open Data](https://github.com/statsbomb/open-data).

Place raw data files in a `data/` directory (created locally, not tracked by git).

---

## Requirements

See [`requirements.txt`](./requirements.txt) for the full list of dependencies.

---

## Authors

Developed as part of **Machine Learning Project at AGH University of Kraków** — by Paweł Lejczak and Jakub Gałka

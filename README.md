# 1. Dimensionality Reduction API
This project provides various dimensionality reduction techniques using a Python-based API. The Conda environment file (`dr_env.yml`) ensures consistent and reproducible dependencies.


## Features
- **Supports 40 DR techniques**: Includes popular methods like PCA, t-SNE, UMAP, and techniques using Tapkee and drtoolbox.
- **Seamless integration**: Combines Python-based, WSL-based, and Octave-based DR methods.

## Requirements
- **Conda Environment**: The project runs in a Conda virtual environment named `dr_api`.
- **Dependencies**:
  - Octave 4.4.1
  - Python libraries (e.g., `umap-learn`, `scikit-learn`, `tensorflow`)
  - WSL (for Tapkee execution)

## Development and Testing Environment
This project was developed and tested in the following environment:
- **Operating System:** Windows 10
- **Python Version:** 3.10.13
- **Conda Version:** conda 24.11.0
- **WSL Version:** WSL2
- **Octave Version:** 4.4.1
  
## Setting up the Conda Environment

### Step 1: Clone the Repository

   ```bash
   git clone https://github.com/JiyeonBae/dr-api.git
   ```

### Step 2: Create and Activate the Environment
- Create the Conda environment using the provided YAML file and activate the newly created environment:
   ```bash
   conda env create -f dr_env.yml
   conda activate dr_api
   ```
- This will create a Conda environment and automatically install all the required libraries and dependencies specified in the `dr_env.yml` file. The YAML file contains a list of necessary packages, their versions, and the channels to install from. Once the environment is activated, you'll have access to all the required libraries for the project.

### Step 3: Install WSL for Using Tapkee Prebuilt Files
Run the following command to install Windows Subsystem for Linux (WSL)
   ```bash
  wsl --install
   ```
### Step 4: Install Octave 4.4.1
- Download and install from: https://ftp.gnu.org/gnu/octave/windows/

### Step 5: Configure Octave Path

**Set the Octave Path in `_drtoolbox.py`**  
   - Open the file `_drtoolbox.py`.
   - Locate the following line:
     ```python
     octave_path = "C:/Octave/Octave-4.4.1/bin/octave.bat"
     ```
   - Replace the path with the location of your Octave installation directory if it differs from the above example.

     
## Hyperparameters and Configuration

The hyperparameters required for each Dimensionality Reduction technique are stored in the `_metadata.json` file. 
This file includes **Recommended Ranges** and **Default Values**.

## Usage Example
After setting up the environment, you can use the core functions from the `dr.py` file as follows:
```python
from sklearn.datasets import load_iris
from dr import *

# Load your dataset (here we're using the Iris dataset as an example)
iris = load_iris()
X = iris.data

# Testing the UMAP function
umap_result = run_umap(X, n_neighbors=5, min_dist=0.1, init="random")
print("UMAP Result:", umap_result)

# Testing the PaCMAP function
pacmap_result = run_pacmap(X, n_neighbors=5, MN_ratio=0.5, FP_ratio=0.5, init="random")
print("\nPaCMAP Result:", pacmap_result)

# The results will be the data with shape (n_samples, 2).
```

# 2. DR Metric Selection
A workflow for reducing bias in metric selection for benchmarking dimensionality reduction techniques, using empirical clustering to select sets of evaluation metrics. Running the Python scripts in the order listed below will reproduce our entire workflow for metric selection and benchmarking.

## Step 1: Projection Generator (`projection.py`)

`projection.py` serves as the experimental driver.  
For each of the **96 datasets** in `sampled_datasets`, the script generates **300 projections** by

1. **Randomly selecting** one of the **40 dimensional-reduction techniques** implemented in `dr.py`; and  
2. **Sampling each method’s hyper-parameters** uniformly within the ranges defined in **`_metadata.json`**.

The resulting embeddings are stored in dataset-specific sub-directories **under the `projections/` folder**, providing a large and diverse benchmark.

## Step 2: Evaluation (`evaluation.py`)
`evaluation.py` scores every generated projection with **40 evaluation metrics** in open-source ZADU library (v 0.1.1; https://github.com/hj-n/zadu.git).

| Stage | Description |
|-------|-------------|
| **1. Input** | • Scans `sampled_datasets/` to locate each high-dimensional dataset (`data.npy`, `label.npy`).<br>• For the matching dataset folder in `projections/`, iterates over all `projection_*.npy` files. |
| **2. Metric Computation** | • Uses **ZADU** (`zadu.ZADU`) with a **spec** list covering local, cluster-level, and global measures.<br>• Computes scores in **batch**.
| **3. Composite Metrics** | • Derives six F-scores on the fly:<br>  `t&c_10`, `t&c_100`, `ca_t&c_10`, `ca_t&c_100`, `s&c_50`, `label_t&c`  = `2·(P·R)/(P+R)` for their respective trustworthiness/continuity or steadiness/cohesiveness pairs. |
| **4. Metric-name Canonicalisation** | • Maps raw ZADU IDs (e.g. `kl_div`, `nh`) to human-readable names (`kl_divergence_σ`, `neighborhood_hit_k`).<br>• Saves the full, final list once to `evaluations/metric_names.npy` and `metric_names.csv` for downstream use. |
| **5. Result Storage** | • For each projection _i_, writes<br>  `evaluations/<dataset>/evaluations_<i>.npy` — the ordered score vector.<br>• Per-dataset logs are written to `projection_logs/<dataset>.log`. |

## Step 3: Correlation Analyzer (`correlation.py`)

This script builds a **Spearman** similarity matrix, where each row and column correspond to an evaluation metric, and visualizes the matrix as a heatmap for every dataset.

| Stage | Description |
|-------|-------------|
| **1. Input** | • Loads the canonical metric order from `evaluations/metric_names.csv`. <br>• Iterates over each dataset directory in **`evaluations/`** and ingests all `evaluations_<i>.npy` files (300 per dataset). |
| **2. Data Assembly** | • Stacks the 300 evaluation vectors into a `(N_metrics × 300)`  **DataFrame**. |
| **3. Correlation Computation** | • Computes a **Spearman-rank correlation** matrix (`34 × 34`) across metrics. |
| **4. Output** | For each dataset `<D>` two artefacts are written to `correlation/<D>/`: <br>• `correlation_matrix.csv` – full numeric matrix. <br>• `correlation_heatmap.png` – colour-coded heat-map (Seaborn, `coolwarm` palette, 300 dpi). |

## Step 4: Clustering & Representative Selection (`clustering.py`)

This script **clusters the average correlation matrix** and extracts a compact set of representative metrics for each desired cluster count.

| Stage | Description |
|-------|-------------|
| **1. Input** | • Reads the **dataset-averaged Spearman correlation matrix** (`correlation/average_correlation_matrix.csv`). |
| **2. Hierarchical Clustering** | • Converts correlation → distance ( 1 – ρ ), builds an **average-linkage dendrogram**, and re-orders metrics (`leaves_list`). |
| **3. Cluster Exploration** | For every cluster count **k=2~10**:<br> 1. Draws a **clustered heat-map** (`clustered_correlation_heatmap_k.png`) with black rectangles outlining clusters.<br> 2. Computes each cluster’s **representative metric** as the member with the highest mean correlation.<br> 3. Saves representative lists to `cluster_representatives.csv`. |
| **4. Measure-Scope Highlighting** | • Axis labels are colour-boxed by scope—local (pink), cluster-level (light-green), global (light-blue)—aiding visual inspection. |

> **Outputs (in `clustering/`)**  
> • `heatmap_before_clustering.png` – baseline 34 × 34 heatmap  
> • `clustered_correlation_heatmap_2-10.png` – heat-maps with cluster blocks & coloured scopes  
> • `cluster_representatives.csv` – table of representative metrics for each k  
> • `dendrogram.png` – hierarchical tree of metric similarities

# Acknowledgments

I sincerely thank the authors of the following repositories for providing foundational implementations for various Dimensionality Reduction techniques and their hyperparameter ranges:

1. **Dimensionality Reduction techniques and the range of Hyperparameters**  
   Repository: [hj-n/umato_exp](https://github.com/hj-n/umato_exp/blob/master/_final_exp/_dr_provider.py)

2. **Dimensionality Reduction techniques, including Tapkee build files and drtoolbox**  
   Repository: [mespadoto/proj-quant-eval](https://github.com/mespadoto/proj-quant-eval/blob/master/code/01_data_collection/projections.py)


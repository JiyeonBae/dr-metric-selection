# Dimensionality Reduction API

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


## Acknowledgments

I sincerely thank the authors of the following repositories for providing foundational implementations for various Dimensionality Reduction techniques and their hyperparameter ranges:

1. **Dimensionality Reduction techniques and the range of Hyperparameters**  
   Repository: [hj-n/umato_exp](https://github.com/hj-n/umato_exp/blob/master/_final_exp/_dr_provider.py)

2. **Dimensionality Reduction techniques, including Tapkee build files and drtoolbox**  
   Repository: [mespadoto/proj-quant-eval](https://github.com/mespadoto/proj-quant-eval/blob/master/code/01_data_collection/projections.py)


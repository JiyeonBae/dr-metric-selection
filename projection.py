import os
import numpy as np
import json
import random
import dr
import logging
import time
import threading

import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
random.seed(None)

# Projection generation pipeline for dimensionality reduction methods
functions = {
    "umap": dr.run_umap,
    "pacmap": dr.run_pacmap,
    "trimap": dr.run_trimap,
    "tsne": dr.run_tsne,
    "umato": dr.run_umato,
    "pca": dr.run_pca,
    "mds": dr.run_mds,
    "isomap": dr.run_isomap,
    "lle": dr.run_lle,
    "lamp": dr.run_lamp,
    "lmds": dr.run_lmds,
    "ae": dr.run_ae,
    "fa": dr.run_fa,
    "fica": dr.run_fica,
    "grp": dr.run_grp,
    "hlle": dr.run_hlle,
    "ipca": dr.run_ipca,
    "kpcapol": dr.run_kpcapol,
    "kpcarbf": dr.run_kpcarbf,
    "kpcasig": dr.run_kpcasig,
    "ltsa": dr.run_ltsa,
    "le": dr.run_le,
    "mlle": dr.run_mlle,
    "nmf": dr.run_nmf,
    "spca": dr.run_spca,
    "srp": dr.run_srp,
    "tsvd": dr.run_tsvd,
    "dm": dr.run_dm,
    "lpp": dr.run_lpp,
    "tapkee_lmds": dr.run_tapkee_lmds,
    "lltsa": dr.run_lltsa,
    "spe": dr.run_spe,
    "ppca": dr.run_ppca,
    "gda": dr.run_gda,
    "mcml": dr.run_mcml,
    "llc": dr.run_llc,
    "lmnn": dr.run_lmnn,
    "mc": dr.run_mc,
    "gplvm": dr.run_gplvm,
    "lmvu": dr.run_lmvu
}

with open('_metadata.json', 'r') as file:
    json_data = json.load(file)

class TimeoutException(Exception):
    pass

def run_with_timeout(func, *args, timeout=1800, **kwargs):
    """
    Execute func with a timeout using threading.
    """
    result_container = {"result": None}

    def target():
        try:
            result_container["result"] = func(*args, **kwargs)
        except Exception as e:
            result_container["result"] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        thread.join(0)
        raise TimeoutException(f"Function {func.__name__} timed out after {timeout} seconds.")

    return result_container["result"]  


def configure_logging(dataset_name):
    log_dir = "./projection_logs"
    """
    Set up logging to file and console for a given dataset.
    """ 
    os.makedirs(log_dir, exist_ok=True) 
    log_file = os.path.join(log_dir, f"{dataset_name}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def random_hyperparameters(bounds, failed_params, X_shape=None):
    """
    Generate random hyperparameters within given bounds.
    """
    params = {}
    for param, range_ in bounds.items():
        if param == "hub_num" and X_shape: 
            max_hub_num = min(range_[1], X_shape[0] - 1) 
            min_hub_num = range_[0]
            if min_hub_num >= max_hub_num:
                min_hub_num = 1
                max_hub_num = max(1, X_shape[0] - 1)
            value = random.randint(min_hub_num, max_hub_num)
            params[param] = value
        elif param == "n_neighbors" and X_shape: 
            max_neighbors = min(range_[1], X_shape[0] - 1) 
            min_neighbors = range_[0]
            if min_neighbors >= max_neighbors:
                min_neighbors = 1
                max_neighbors = max(1, X_shape[0] - 1)
            value = random.randint(min_neighbors, max_neighbors)
            params[param] = value
        elif isinstance(range_[0], int):
            value = random.randint(range_[0], range_[1])
            params[param] = value
        elif isinstance(range_[0], float):
            value = random.uniform(range_[0], range_[1])
            params[param] = value
        elif isinstance(range_, list):
            value = random.choice(range_)
            params[param] = value
        else:
            raise TypeError(f"Unsupported range type for parameter '{param}': {range_}")
    return params


def adjust_pacmap_bounds(bounds, X_shape, threshold):
    """
    Narrow PACMAP bounds when dataset size is below threshold.
    """

    adjusted_bounds = bounds.copy()
    sample_size = X_shape[0]

    if sample_size < threshold:
        max_mid_near_ratio = (sample_size - 1) / sample_size
        adjusted_bounds["MN_ratio"] = [
            bounds.get("MN_ratio", [0.1, 5.0])[0], 
            min(bounds.get("MN_ratio", [0.1, 5.0])[1], max_mid_near_ratio)
        ]

    return adjusted_bounds


def generate_projections(X, n_projections, save_path, max_retries=3):
    """
    Generate multiple DR projections with retries and timeouts.
    """
    projections = []
    failed_methods = set()
    excluded_methods = set()
    fixed_hyperparameters = {}
    failed_kernels_per_dataset = {}

    i = 0
    while i < n_projections:
        retries = 0
        successful = False
        current_method = None

        while retries < max_retries:
            if retries == 0 or current_method is None:

                available_methods = [
                    key for key in functions.keys()
                    if key not in failed_methods and key not in excluded_methods
                ]

                if not available_methods:
                    logging.warning("No available methods to select from. Stopping projection generation.")
                    return projections

                current_method = random.choice(available_methods)

            try:
                bounds = json_data[current_method]["bounds"]
                if current_method == "pacmap":
                    if X.shape[0] < 400:
                        bounds = adjust_pacmap_bounds(bounds, X.shape, threshold=400)
                    else:
                        logging.info(f"Dataset size ({X.shape[0]}) : Using original bounds for PACMAP.")
                if current_method in fixed_hyperparameters:
                    hyperparameters = fixed_hyperparameters[current_method]
                    logging.info(f"Using fixed hyperparameters for method {current_method}: {hyperparameters}")
                else:
                    hyperparameters = random_hyperparameters(bounds, set(), X.shape)

                if current_method == "gda":
                    failed_kernels = failed_kernels_per_dataset.get(id(X), set())
                    kernel_bounds = bounds.get("kernel", [])
                    valid_kernels = [k for k in kernel_bounds if k not in failed_kernels]
                    if not valid_kernels:
                        logging.warning(f"No valid kernels left for gda with dataset {id(X)}.")
                        failed_methods.add(current_method)
                        break
                    selected_kernel = random.choice(valid_kernels)
                    hyperparameters["kernel"] = selected_kernel

                logging.info(f"Selected method: {current_method}, Hyperparameters: {hyperparameters}")

                start_time = time.time()
                result = run_with_timeout(functions[current_method], X, timeout=1800, **hyperparameters)
                elapsed_time = time.time() - start_time

                if result is None or np.isnan(result).any() or result.size == 0:
                    logging.error(f"Method {current_method} generated invalid projection (NaN or empty result). Retrying...")
                    retries += 1
                    if current_method == "gda":
                        failed_kernels_per_dataset.setdefault(id(X), set()).add(hyperparameters["kernel"])
                    continue

                projection_save_path = os.path.join(save_path, f"projection_{i}.npy")
                meta_save_path = os.path.join(save_path, f"projection_{i}_meta.json")
                np.save(projection_save_path, result)
                with open(meta_save_path, "w") as meta_file:
                    json.dump({"method": current_method, "hyperparameters": hyperparameters}, meta_file, indent=4)

                logging.info(f"Saved projection {i + 1}/{n_projections}: {projection_save_path}")
                projections.append((current_method, hyperparameters, result))
                successful = True
                break
            
            except TimeoutException as e:
                logging.warning(f"Timeout occurred: {e}")
                excluded_methods.add(current_method)
                current_method = None
                break

            except ValueError as e:
                logging.error(f"ValueError in method: {current_method}, Hyperparameters: {hyperparameters}, Error: {e}")
                retries += 1
                logging.warning(f"Retry {retries}/{max_retries} for method {current_method}.")
                if current_method == "gda":
                    failed_kernels_per_dataset.setdefault(id(X), set()).add(hyperparameters.get("kernel", None))

            except Exception as e:
                logging.error(f"General Error in method: {current_method}, Hyperparameters: {hyperparameters}, Error: {e}")
                retries += 1
                logging.warning(f"Retry {retries}/{max_retries} for method {current_method}.")
                if current_method == "gda":
                    failed_kernels_per_dataset.setdefault(id(X), set()).add(hyperparameters.get("kernel", None))

        if successful:
            i += 1
        else:
            logging.warning(f"Failed for projection {i} with method {current_method}.")
            if retries >= max_retries:
                failed_methods.add(current_method)

    return projections
    

# Process all datasets in the specified directory
def process_all_datasets(sampled_dataset_dir, n_projections, save_dir="./projections"):
    """
    Process every dataset folder, generating projections and saving results.
    """
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    datasets = [name for name in os.listdir(sampled_dataset_dir) if os.path.isdir(os.path.join(sampled_dataset_dir, name))]
    for dataset_name in datasets:
        try:
            logging.info(f"Processing dataset: {dataset_name}")

            dataset_path = os.path.join(sampled_dataset_dir, dataset_name)
            data = np.load(os.path.join(dataset_path, "data.npy"))
            label = np.load(os.path.join(dataset_path, "label.npy"))

            projection_save_path = os.path.join(save_dir, dataset_name)
            os.makedirs(projection_save_path, exist_ok=True)

            generate_projections(data, n_projections=n_projections, save_path=projection_save_path)

            logging.info(f"Finished processing dataset: {dataset_name}")

        except Exception as e:
            logging.error(f"Error processing dataset {dataset_name}: {e}")


# Process a single dataset by its index in the directory listing
def process_specific_dataset(sampled_dataset_dir, dataset_index, n_projections, save_dir="./projections"):
    """
    Process only the dataset at dataset_index, using logging per dataset.
    """

    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    datasets = [name for name in os.listdir(sampled_dataset_dir) if os.path.isdir(os.path.join(sampled_dataset_dir, name))]

    if dataset_index < 0 or dataset_index >= len(datasets):
        print(f"Invalid dataset index: {dataset_index}. Must be between 0 and {len(datasets) - 1}.")
        return

    dataset_name = datasets[dataset_index]
    configure_logging(dataset_name)

    try:
        logging.info(f"Processing dataset: {dataset_name}")

        dataset_path = os.path.join(sampled_dataset_dir, dataset_name)
        data = np.load(os.path.join(dataset_path, "data.npy"))
        label = np.load(os.path.join(dataset_path, "label.npy"))

        projection_save_path = os.path.join(save_dir, dataset_name)
        os.makedirs(projection_save_path, exist_ok=True)

        failed_methods = set()

        logging.info(f"Dataset size : {data.shape}")
        generate_projections(data, n_projections=n_projections, save_path=projection_save_path)

        logging.info(f"Finished processing dataset: {dataset_name}")

    except Exception as e:
        logging.error(f"Error processing dataset {dataset_name}: {e}")


# Process datasets starting from a given index up to the end
def process_datasets_from_index(sampled_dataset_dir, start_index, n_projections, save_dir="./projections"):
    """
    Process datasets sequentially from start_index to the last one.
    """
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    datasets = [name for name in os.listdir(sampled_dataset_dir) if os.path.isdir(os.path.join(sampled_dataset_dir, name))]

    if start_index < 0 or start_index >= len(datasets):
        print(f"Invalid start index: {start_index}. Must be between 0 and {len(datasets) - 1}.")
        return

    for dataset_index in range(start_index, len(datasets)):
        process_specific_dataset(sampled_dataset_dir, dataset_index, n_projections, save_dir)


if __name__ == "__main__":
    # Choose one of the following processing functions:
    # 1) process_all_datasets: process every dataset in the directory
    # 2) process_specific_dataset: process only the dataset at a given index
    # 3) process_datasets_from_index: process datasets starting from a given index
    # By default, we run process_all_datasets for all datasets
    SAMPLE_DIR = "./sampled_datasets"
    N_PROJECTIONS = 300     # number of projections to generate
    process_all_datasets(SAMPLE_DIR, N_PROJECTIONS)

import os
import numpy as np
import logging
from zadu import zadu

DATASET_DIR = os.path.abspath("./sampled_datasets")
PROJECTION_DIR = os.path.abspath("./projections")
RESULTS_DIR = os.path.abspath("./evaluations")
os.makedirs(RESULTS_DIR, exist_ok=True)

metric_names_global_saved = False

spec = [
    {"id": "tnc", "params": {"k": 10}},
    {"id": "tnc", "params": {"k": 100}},
    {"id": "mrre", "params": {"k": 10}},
    {"id": "mrre", "params": {"k": 100}},
    {"id": "lcmc", "params": {"k": 10}},
    {"id": "lcmc", "params": {"k": 100}},
    {"id": "nh", "params": {"k": 10}},
    {"id": "nh", "params": {"k": 100}},
    {"id": "nd", "params": {"k": 10}},
    {"id": "nd", "params": {"k": 100}},
    {"id": "ca_tnc", "params": {"k": 10}},
    {"id": "ca_tnc", "params": {"k": 100}},
    {"id": "proc", "params": {"k": 10}},
    {"id": "proc", "params": {"k": 100}},
    {"id": "snc", "params": {"iteration": 150, "walk_num_ratio": 0.3, "alpha": 0.1, "k": 50, "clustering_strategy": "dbscan"}},
    {"id": "dsc", "params": {}},
    {"id": "ivm", "params": {"measure": "silhouette"}},
    {"id": "c_evm", "params": {"measure": "arand", "clustering": "kmeans", "clustering_args": None}},
    {"id": "l_tnc", "params": {"cvm": "dsc"}},
    {"id": "stress", "params": {}},
    {"id": "kl_div", "params": {"sigma": 0.1}},
    {"id": "kl_div", "params": {"sigma": 1}},
    {"id": "dtm", "params": {"sigma": 0.1}},
    {"id": "dtm", "params": {"sigma": 1}},
    {"id": "pr", "params": {}},
    {"id": "srho", "params": {}} 
]

def compute_f1_score(val1, val2):
    # Compute the F1-score
    if val1 + val2 == 0:
        return 0.0
    return 2 * (val1 * val2) / (val1 + val2)

def insert_new_metrics(scores, metric_names):
    # Mapping from metric names to their index values
    index_map = {name: i for i, name in enumerate(metric_names)}

    # Add t&c metrics
    insert_positions = {
        "t&c_10": index_map["trustworthiness_10"] + 2,
        "t&c_100": index_map["trustworthiness_100"] + 2,
        "ca_t&c_10": index_map["ca_trustworthiness_10"] + 2,
        "ca_t&c_100": index_map["ca_trustworthiness_100"] + 2,
        "s&c_50": index_map["steadiness_50"] + 2,
        "label_t&c": index_map["label_trustworthiness"] + 2,
    }
    
    new_scores = {
        "t&c_10": compute_f1_score(scores[index_map["trustworthiness_10"]], scores[index_map["continuity_10"]]),
        "t&c_100": compute_f1_score(scores[index_map["trustworthiness_100"]], scores[index_map["continuity_100"]]),
        "ca_t&c_10": compute_f1_score(scores[index_map["ca_trustworthiness_10"]], scores[index_map["ca_continuity_10"]]),
        "ca_t&c_100": compute_f1_score(scores[index_map["ca_trustworthiness_100"]], scores[index_map["ca_continuity_100"]]),
        "s&c_50": compute_f1_score(scores[index_map["steadiness_50"]], scores[index_map["cohesiveness_50"]]),
        "label_t&c": compute_f1_score(scores[index_map["label_trustworthiness"]], scores[index_map["label_continuity"]]),
    }
    
    updated_scores = list(scores)
    updated_metric_names = list(metric_names)
    for key, pos in sorted(insert_positions.items(), key=lambda x: x[1], reverse=True):
        updated_scores.insert(pos, new_scores[key])
        updated_metric_names.insert(pos, key)
    return np.array(updated_scores), updated_metric_names

dataset_list = sorted(os.listdir(DATASET_DIR))
final_metric_names = None

for dataset_name in dataset_list:
    dataset_path = os.path.join(DATASET_DIR, dataset_name)
    projection_path = os.path.join(PROJECTION_DIR, dataset_name)
    dataset_results_path = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(dataset_results_path, exist_ok=True)

    if not os.path.isdir(dataset_path):
        continue

    configure_logging(dataset_name)

    try:
        logging.info(f"Processing dataset: {dataset_name}")

        hd = np.load(os.path.join(dataset_path, "data.npy"))
        labels = np.load(os.path.join(dataset_path, "label.npy"))
        assert len(labels) == hd.shape[0], \
            f"The size of the labels array does not match the number of samples in {dataset_name}."

        projection_files = sorted(
            [f for f in os.listdir(projection_path)
             if f.endswith(".npy") and f.startswith("projection_")],
            key=lambda x: int(x.replace("projection_", "").replace(".npy", ""))
        )

        for projection_file in projection_files:
            ld = np.load(os.path.join(projection_path, projection_file))
            if np.isnan(ld).any():
                logging.warning(f"Projection {projection_file} contains NaN values.")

            try:
                scores = zadu.ZADU(spec, hd).measure(ld, label=labels)

                metric_names = []
                structured_scores = []
                for metric, score in zip(spec, scores):
                    params = metric['params']
                    mid = metric['id']

                    if isinstance(score, dict):
                        for key, value in score.items():
                            if 'k' in params:
                                name = f"{key}_{params['k']}"
                            elif 'sigma' in params:
                                name = f"{key}_{params['sigma']}"
                            else:
                                name = key
                            metric_names.append(name)
                            structured_scores.append(value)

                    else:
                        if mid == 'ivm':
                            base = params['measure']  # silhouette
                        elif mid == 'c_evm':
                            base = f"{params['clustering']}_{params['measure']}" # kmeans_arand
                        elif mid == 'dsc':
                            base = 'distance_consistency'
                        elif mid == 'proc':
                            base = 'procrustes'
                        elif mid in ('nh', 'nd'):
                            base = {'nh':'neighborhood_hit','nd':'neighbor_dissimilarity'}[mid]
                        elif mid == 'kl_div':
                            base = 'kl_divergence'
                        elif mid == 'dtm':
                            base = 'distance_to_measure'
                        elif mid == 'pr':
                            base = 'pearson_r'
                        elif mid == 'srho':
                            base = 'spearman_rho'
                        else:
                            base = mid

                        if 'k' in params:
                            name = f"{base}_{params['k']}"
                        elif 'sigma' in params:
                            name = f"{base}_{params['sigma']}"
                        else:
                            name = base

                        metric_names.append(name)
                        structured_scores.append(score)

                if final_metric_names is None:
                    dummy, final_metric_names = insert_new_metrics(np.array(structured_scores), metric_names)
                    # Save metric_names as .npy and .csv files
                    np.save(os.path.join(RESULTS_DIR, "metric_names.npy"), np.array(final_metric_names))
                    with open(os.path.join(RESULTS_DIR, "metric_names.csv"), "w") as f:
                        f.write(",".join(final_metric_names))
                    metric_names_global_saved = True

                # Insert the F1-score metric
                updated_scores, _ = insert_new_metrics(np.array(structured_scores), metric_names)
                base_idx = int(projection_file.replace("projection_", "").replace(".npy", ""))
                result_file = os.path.join(dataset_results_path, f"evaluations_{base_idx}.npy")
                np.save(result_file, updated_scores)
                logging.info(f"Saved evaluation results to: {result_file}")

            except Exception as e:
                logging.error(f"Error processing projection {projection_file}: {e}")

    except Exception as e:
        logging.error(f"Error processing dataset {dataset_name}: {e}")

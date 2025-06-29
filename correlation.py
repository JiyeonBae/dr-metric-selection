import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

evaluations3_dir = os.path.abspath("./evaluations")
correlation_dir = os.path.abspath("./correlation")

os.makedirs(correlation_dir, exist_ok=True)

def load_metric_names():
    metric_names_path = os.path.join(evaluations3_dir, 'metric_names.csv')
    if os.path.exists(metric_names_path):
        return pd.read_csv(metric_names_path, header=None).iloc[:, 0].str.strip().tolist()
    else:
        raise FileNotFoundError(f"Metric names file not found at {metric_names_path}")

def generate_correlation_heatmaps():
    dataset_list = sorted([d for d in os.listdir(evaluations3_dir) if os.path.isdir(os.path.join(evaluations3_dir, d))])
    
    try:
        metric_names = load_metric_names()
    except FileNotFoundError as e:
        print(e)
        return
    
    for dataset_name in dataset_list:
        dataset_path = os.path.join(evaluations3_dir, dataset_name)
        correlation_path = os.path.join(correlation_dir, dataset_name)
        os.makedirs(correlation_path, exist_ok=True)
        
        data_list = []
        for i in range(300):
            file_path = os.path.join(dataset_path, f'evaluations_{i}.npy')
            if os.path.exists(file_path):
                evaluation = np.load(file_path)
                
                if evaluation.ndim == 1 and evaluation.shape[0] == len(metric_names):
                    data_list.append(evaluation)
                else:
                    print(f"⚠ Invalid shape in {file_path}: Expected (34,), got {evaluation.shape}")

        if not data_list:
            print(f"⚠ No valid data found in {dataset_name}. Skipping.")
            continue

        # DataFrame of shape (34, N)
        evaluations_df = pd.DataFrame(data_list, columns=metric_names).transpose()

        print(f"DataFrame Shape for {dataset_name}: {evaluations_df.shape}")

        if evaluations_df.shape[0] != len(metric_names):
            print(f"⚠ ERROR: DataFrame row count {evaluations_df.shape[0]} does not match metric count {len(metric_names)}")
            continue

        # Transpose the 34x34 correlation matrix
        correlation_matrix = evaluations_df.T.corr(method='spearman')

        print(f"Correlation Matrix Shape: {correlation_matrix.shape}")

        # Save the heatmap
        correlation_matrix_path = os.path.join(correlation_path, 'correlation_matrix.csv')
        correlation_heatmap_path = os.path.join(correlation_path, 'correlation_heatmap.png')

        correlation_matrix.to_csv(correlation_matrix_path)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5,
            xticklabels=metric_names, yticklabels=metric_names, square=True, cbar=False
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(correlation_heatmap_path, dpi=300)
        plt.close()
        
        print(f"Saved Spearman heatmap and correlation matrix for {dataset_name}")

generate_correlation_heatmaps()


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster, dendrogram
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, Patch

correlation_dir = os.path.abspath("./correlation")
clustering_dir = os.path.join("./clustering")
os.makedirs(clustering_dir, exist_ok=True)

avg_correlation_path = os.path.join(correlation_dir, 'average_correlation_matrix.csv')
avg_valid_correlation_df = pd.read_csv(avg_correlation_path, index_col=0)

# Categorization of Measures by Scope
local_measures = {
    "trustworthiness_10", "continuity_10", "t&c_10", "trustworthiness_100", "continuity_100", "t&c_100",
    "mrre_false_10", "mrre_missing_10", "mrre_false_100", "mrre_missing_100",
    "lcmc_10", "lcmc_100", "neighborhood_hit_10", "neighborhood_hit_100",
    "neighbor_dissimilarity_10", "neighbor_dissimilarity_100",
    "ca_trustworthiness_10", "ca_continuity_10", "ca_t&c_10",
    "ca_trustworthiness_100", "ca_continuity_100", "ca_t&c_100",
    "procrustes_10", "procrustes_100"
}

cluster_measures = {
    "steadiness_50", "cohesiveness_50", "s&c_50", "distance_consistency", "silhouette",
    "kmeans_arand", "label_trustworthiness", "label_continuity", "label_t&c"
}

global_measures = {
    "stress", "kl_divergence_0.1", "kl_divergence_1", "distance_to_measure_0.1",
    "distance_to_measure_1", "pearson_r", "spearman_rho"
}

# Heatmap of the correlation matrix before clustering
plt.figure(figsize=(12, 10))
custom_cmap = sns.color_palette("coolwarm", as_cmap=True)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
ax = sns.heatmap(
    avg_valid_correlation_df,
    annot=False,
    cmap=custom_cmap,
    norm=norm,
    linewidths=0.5,
    xticklabels=avg_valid_correlation_df.index,
    yticklabels=avg_valid_correlation_df.index,
    square=True,
    cbar=True
)
plt.title("Heatmap Before Clustering")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
heatmap_before_path = os.path.join(clustering_dir, "heatmap_before_clustering.png")
plt.savefig(heatmap_before_path, dpi=300)
plt.close()
print(f"Heatmap of the correlation matrix before clustering has been saved: {heatmap_before_path}")

# Reordering based on hierarchical clustering
distance_matrix = 1 - avg_valid_correlation_df
condensed_distance = squareform(distance_matrix, checks=False)
Z = linkage(condensed_distance, method='average')
cluster_indices = leaves_list(Z)

# Dictionary to store representative elements for each cluster count
all_representatives_dict = {}

max_clusters = min(len(avg_valid_correlation_df), 10)

for n_clusters in range(2, max_clusters + 1):
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Save heatmap
    plt.figure(figsize=(12, 10))
    custom_cmap = sns.color_palette("coolwarm", as_cmap=True)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    ax = sns.heatmap(
        avg_valid_correlation_df.iloc[cluster_indices, cluster_indices],
        annot=False,
        cmap=custom_cmap,
        norm=norm,
        linewidths=0.5,
        xticklabels=avg_valid_correlation_df.index[cluster_indices],
        yticklabels=avg_valid_correlation_df.index[cluster_indices],
        square=True,
        cbar=True
    )

    # Calculate representative element for each cluster
    cluster_boundaries = {}
    for i, lbl in enumerate(cluster_labels[cluster_indices]):
        if lbl not in cluster_boundaries:
            cluster_boundaries[lbl] = [i, i]
        else:
            cluster_boundaries[lbl][1] = i

    rep_list = []
    for cluster_num, (start, end) in cluster_boundaries.items():
        rect = Rectangle(
            (start, start),
            end - start + 1,
            end - start + 1,
            linewidth=3,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)
        members = cluster_indices[start:end + 1]
        cluster_corr_matrix = avg_valid_correlation_df.iloc[members, members]
        avg_correlations = cluster_corr_matrix.mean(axis=1)
        if avg_correlations.dropna().empty:
            representative_item = None
        else:
            representative_item = avg_correlations.idxmax()
        if representative_item is not None:
            rep_list.append(representative_item)

    rep_list_unique = list(dict.fromkeys(rep_list))
    all_representatives_dict[n_clusters] = rep_list_unique

    # Highlight cluster boundaries
    cluster_boundaries = {}
    for i, lbl in enumerate(cluster_labels[cluster_indices]):
        if lbl not in cluster_boundaries:
            cluster_boundaries[lbl] = [i, i]
        else:
            cluster_boundaries[lbl][1] = i
    for _, (start, end) in cluster_boundaries.items():
        rect = Rectangle(
            (start, start),
            end - start + 1,
            end - start + 1,
            linewidth=3,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)

    for label in ax.get_xticklabels():
        text = label.get_text()
        if text in local_measures:
            label.set_bbox(dict(facecolor="pink", edgecolor="none", alpha=0.5, boxstyle='round,pad=0.2'))
        elif text in cluster_measures:
            label.set_bbox(dict(facecolor="lightgreen", edgecolor="none", alpha=0.5, boxstyle='round,pad=0.2'))
        elif text in global_measures:
            label.set_bbox(dict(facecolor="lightblue", edgecolor="none", alpha=0.5, boxstyle='round,pad=0.2'))
    for label in ax.get_yticklabels():
        text = label.get_text()
        if text in local_measures:
            label.set_bbox(dict(facecolor="pink", edgecolor="none", alpha=0.5, boxstyle='round,pad=0.2'))
        elif text in cluster_measures:
            label.set_bbox(dict(facecolor="lightgreen", edgecolor="none", alpha=0.5, boxstyle='round,pad=0.2'))
        elif text in global_measures:
            label.set_bbox(dict(facecolor="lightblue", edgecolor="none", alpha=0.5, boxstyle='round,pad=0.2'))

    legend_patches = [
        Patch(facecolor="pink", edgecolor="black", label="Local Measures"),
        Patch(facecolor="lightgreen", edgecolor="black", label="Cluster-level Measures"),
        Patch(facecolor="lightblue", edgecolor="black", label="Global Measures")
    ]
    plt.legend(handles=legend_patches, loc="upper right", title="Measure Categories", fontsize=10, frameon=True)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    heatmap_path = os.path.join(clustering_dir, f'clustered_correlation_heatmap_{int(n_clusters)}.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"{n_clusters} clusters: {heatmap_path}")

# Save representative elements as a CSV file
rows = []
for n_clusters, rep_list in all_representatives_dict.items():
    reps_str = ", ".join(rep_list)
    rows.append({"n_clusters": n_clusters, "Representatives": reps_str})

representatives_df = pd.DataFrame(rows)
representatives_csv_path = os.path.join(clustering_dir, "cluster_representatives.csv")
representatives_df.to_csv(representatives_csv_path, index=False)
print(f"Representative cluster elements saved: {representatives_csv_path}")

# Save dendrogram
plt.figure(figsize=(16, 8))
dendrogram(
    Z,
    labels=avg_valid_correlation_df.index[cluster_indices].tolist(),
    leaf_rotation=45,
    leaf_font_size=14
)
plt.title("Hierarchical Clustering Dendrogram", fontsize=16)
plt.xlabel("Data Points", fontsize=14)
plt.ylabel("Distance", fontsize=14)
plt.xticks(fontsize=12, rotation=45, ha="right")
plt.yticks(fontsize=12)
dendrogram_path = os.path.join(clustering_dir, "dendrogram.png")
plt.savefig(dendrogram_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Dendrogram saved: {dendrogram_path}")

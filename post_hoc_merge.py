import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

folder_path = "./data/"
cluster_data = pd.read_csv(folder_path+'dbscan_cluster_summary.csv', index_col=0)

def merge_clusters(cluster_data, desired_clusters):
    while len(cluster_data.columns) > desired_clusters:
        centroids = cluster_data.mean(axis=0)
        dist_matrix = distance_matrix(centroids.values.reshape(-1, 1), centroids.values.reshape(-1, 1))
        np.fill_diagonal(dist_matrix, np.inf)
        merge_indices = np.unravel_index(np.argmin(dist_matrix, axis=None), dist_matrix.shape)
        merged_cluster = cluster_data.iloc[:, merge_indices[0]] + cluster_data.iloc[:, merge_indices[1]]
        merged_name = f'merged_{cluster_data.columns[merge_indices[0]]}_{cluster_data.columns[merge_indices[1]]}'
        cluster_data[merged_name] = merged_cluster
        cluster_data.drop(columns=[cluster_data.columns[i] for i in merge_indices], inplace=True)
    return cluster_data

merged_clusters = merge_clusters(cluster_data, 50)
merged_clusters.to_csv(folder_path+'merged_cluster_summary.csv', index=True)
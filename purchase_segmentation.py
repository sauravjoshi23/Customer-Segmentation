import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from collections import Counter

def create_user_aisle_matrix(data):
    return pd.crosstab(data['user_id'], data['aisle'])

def scale_data(matrix):
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)

def determine_optimal_clusters(data):
    inertia = []
    for k in range(1, 21): 
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.plot(range(1, 21), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow curve')
    plt.show()

def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

def apply_dbscan(data):
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(data)
    return labels

def interpret_clusters(matrix, labels):
    cluster_info = pd.concat([matrix, pd.Series(labels, name='Cluster', index=matrix.index)], axis=1)
    cluster_summary = cluster_info.groupby('Cluster').mean().transpose()
    return cluster_summary

if __name__ == '__main__':
    folder_path = "./data/"
    data = pd.read_csv(folder_path+'merged_data.csv')
    user_aisle_matrix = create_user_aisle_matrix(data)
    scaled_data = scale_data(user_aisle_matrix)
    # determine_optimal_clusters(scaled_data)
    num_clusters = 2
    kmeans_labels = apply_kmeans(scaled_data, num_clusters)
    dbscan_labels = apply_dbscan(scaled_data)
    kmeans_clusters_summary = interpret_clusters(user_aisle_matrix, kmeans_labels)
    print(kmeans_clusters_summary)
    dbscan_clusters_summary = interpret_clusters(user_aisle_matrix, dbscan_labels)
    print(dbscan_clusters_summary)
    kmeans_clusters_summary.to_csv(folder_path+'kmeans_cluster_summary.csv')
    dbscan_clusters_summary.to_csv(folder_path+'dbscan_cluster_summary.csv')

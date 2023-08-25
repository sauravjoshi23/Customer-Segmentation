import pandas as pd
import numpy as np

folder_path = "./data/"
cluster_summary = pd.read_csv(folder_path+'merged_cluster_summary.csv')
cluster_summary = cluster_summary.apply(pd.to_numeric, errors='coerce')
cluster_sizes = cluster_summary.sum(axis=0)
cluster_data = pd.read_csv(folder_path+'dominant_aisles_per_cluster.csv')
cluster_data['Cluster'] = cluster_data.index
cluster_data['Dominant_Aisles'] = cluster_data[['0', '1', '2', '3', '4']].values.tolist()
cluster_data['campaign'] = "Exclusive deals on top products: " + cluster_data['Dominant_Aisles'].str.join(', ')
cluster_mapping = dict(zip(cluster_sizes.index, range(len(cluster_sizes[1:]))))
cluster_data['Cluster'] = cluster_data['Cluster'].replace(cluster_mapping)
cluster_sizes.index = [cluster_mapping[idx] if idx in cluster_mapping else idx for idx in cluster_sizes.index]
cluster_data['exposed_members_count'] = (0.2 * cluster_sizes).astype(int)
cluster_data[['Cluster', 'campaign', 'exposed_members_count']].to_csv(folder_path+'cluster_campaign_details.csv', index=False)

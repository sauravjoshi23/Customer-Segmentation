import pandas as pd
import numpy as np

folder_path = "./data/"
kmeans_labels = pd.read_csv(folder_path+'kmeans_labels.csv', squeeze=True) 
cluster_sizes = kmeans_labels.value_counts()
cluster_data = pd.read_csv(folder_path+'dominant_aisles_per_cluster.csv')
cluster_data['Cluster'] = cluster_data.index
cluster_data['Dominant_Aisles'] = cluster_data[['0', '1', '2', '3', '4']].values.tolist()
cluster_data['campaign'] = "Exclusive deals on top products: " + cluster_data['Dominant_Aisles'].str.join(', ')
cluster_data['exposed_members_count'] = (0.2 * cluster_sizes).astype(int)
cluster_data[['Cluster', 'campaign', 'exposed_members_count']].to_csv(folder_path+'cluster_campaign_details.csv', index=False)
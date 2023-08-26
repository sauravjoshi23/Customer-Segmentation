import pandas as pd

N = 5
folder_path = "./data/"
cluster_summary = pd.read_csv(folder_path+'kmeans_cluster_summary.csv', index_col=0)
cluster_summary = cluster_summary.apply(pd.to_numeric, errors='coerce')
dominant_aisles = cluster_summary.apply(lambda x: x.nlargest(N).index.tolist(), axis=0)
dominant_aisles = dominant_aisles.T
dominant_aisles.to_csv(folder_path+'dominant_aisles_per_cluster.csv', index=True)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def create_user_aisle_matrix(data):
    return pd.crosstab(data['user_id'], data['aisle'])

def assign_treatment(group):
    treated_indices = np.random.choice(group.index, group['exposed_members_count'].iloc[0], replace=False)
    group['is_treated'] = group.index.isin(treated_indices)
    return group

folder_path = "./data/"
user_aisle_data = pd.read_csv(folder_path+'merged_data.csv')
campaign_data = pd.read_csv(folder_path+'cluster_campaign_details.csv')
cluster_summary = pd.read_csv(folder_path+'kmeans_cluster_summary.csv')

cluster_centers = cluster_summary.set_index('Unnamed: 0').transpose()
missing_columns = [col for col in cluster_centers.columns if col not in user_aisle_matrix.columns]
user_aisle_matrix = create_user_aisle_matrix(user_aisle_data)
for col in missing_columns:
    user_aisle_matrix[col] = 0

user_aisle_matrix = user_aisle_matrix[cluster_centers.columns]
distances = cdist(user_aisle_matrix.values, cluster_centers.values, metric='euclidean')
cluster_assignments = np.argmin(distances, axis=1)

users_cluster = pd.DataFrame({
    'user_id': user_aisle_matrix.index,
    'Cluster': cluster_assignments
})

users_cluster = users_cluster.merge(campaign_data[['Cluster', 'exposed_members_count']], on='Cluster', how='left')
np.random.seed(42)  

users_cluster = users_cluster.groupby('Cluster').apply(assign_treatment)
users_cluster[['user_id', 'Cluster', 'is_treated']].to_csv(folder_path+'user_treatment_assignment.csv', index=False)

users_data_path = folder_path + 'user_treatment_assignment.csv'
users_cluster = pd.read_csv(users_data_path)

X = users_cluster[['Cluster']]
y = users_cluster['is_treated']
model = LogisticRegression()
model.fit(X, y)
users_cluster['propensity_score'] = model.predict_proba(X)[:, 1]
treated = users_cluster[users_cluster['is_treated'] == True]
control = users_cluster[users_cluster['is_treated'] == False]
matched_control_indices = []
matched_to_treated_user_ids = []
for index, row in treated.iterrows():
    differences = abs(control['propensity_score'] - row['propensity_score'])
    closest_index = differences.idxmin()
    matched_control_indices.append(closest_index)
    matched_to_treated_user_ids.append(row['user_id'])
    control = control.drop(closest_index)

matched_control = users_cluster.loc[matched_control_indices]
matched_control['matched_to_treated_user_id'] = matched_to_treated_user_ids
matched_control.to_csv(folder_path+'matched_control_users.csv', index=False)
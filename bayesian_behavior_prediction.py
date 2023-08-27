import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

folder_path = "./data/"
merged_data_path = folder_path+"merged_data.csv"
matched_control_users_path = folder_path+"matched_control_users.csv"
merged_data = pd.read_csv(merged_data_path, index_col='user_id')
users_cluster = pd.read_csv(matched_control_users_path, index_col='user_id')
fresh_fruits_users = merged_data[merged_data['aisle'] == 'fresh fruits'].index.unique()
users_cluster['bought_fresh_fruits'] = users_cluster.index.map(lambda x: 1 if x in fresh_fruits_users else 0)
user_aisle_matrix = pd.crosstab(merged_data.index, merged_data['aisle'])
full_data = user_aisle_matrix.join(users_cluster[['Cluster', 'bought_fresh_fruits']], how='left').fillna(0)
y = full_data['bought_fresh_fruits']
X = full_data.drop(['bought_fresh_fruits', 'fresh fruits'], axis=1, errors='ignore')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
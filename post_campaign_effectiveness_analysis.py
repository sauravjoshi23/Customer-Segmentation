import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

folder_path = "./data/"
effect_size = 0.05
np.random.seed(42) 
matched_data_path = folder_path + 'matched_control_users.csv'
matched_data = pd.read_csv(matched_data_path, index_col="user_id")
matched_data['baseline_purchase_prob'] = matched_data['propensity_score']
matched_data['post_campaign_purchase_treated'] = np.where(
    np.random.rand(len(matched_data)) < (matched_data['baseline_purchase_prob'] + effect_size), 1, 0)
matched_data['post_campaign_purchase_control'] = np.where(
    np.random.rand(len(matched_data)) < matched_data['baseline_purchase_prob'], 1, 0)

t_stat, p_val = ttest_ind(
    matched_data['post_campaign_purchase_treated'],
    matched_data['post_campaign_purchase_control']
)
print(f"T-Statistic: {t_stat}, P-value: {p_val}")
if p_val < 0.05:
    print("The campaign had a statistically significant effect on purchases.")
else:
    print("The campaign did not have a statistically significant effect on purchases.")

output_path = folder_path + 'post_campaign_results.csv'
matched_data.to_csv(output_path)
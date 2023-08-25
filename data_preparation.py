import pandas as pd

def merge_datasets(folder_path, sample_fraction):
    products_df = pd.read_csv(folder_path+'products.csv')
    aisles_df = pd.read_csv(folder_path+'aisles.csv')
    order_products_prior_sample = pd.read_csv(folder_path+'order_products__prior.csv').sample(frac=sample_fraction)
    sampled_order_ids = order_products_prior_sample['order_id'].unique()
    orders_df = pd.read_csv(folder_path+'orders.csv')
    orders_sample = orders_df[orders_df['order_id'].isin(sampled_order_ids)]
    products_with_aisles_df = pd.merge(products_df, aisles_df, on='aisle_id', how='inner')
    merged_df = pd.merge(order_products_prior_sample, products_with_aisles_df, on='product_id', how='inner')
    final_df = pd.merge(merged_df, orders_sample[['order_id', 'user_id']], on='order_id', how='inner')
    final_df = final_df[['user_id', 'product_name', 'aisle']]
    final_df.dropna(inplace=True)
    return final_df

if __name__ == '__main__':
    folder_path = "./data/"
    sample_fraction = 0.02
    result_df = merge_datasets(folder_path, sample_fraction)
    result_df.to_csv(folder_path+'merged_data.csv', index=False)

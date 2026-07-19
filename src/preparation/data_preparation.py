"""
Data Preparation and Preprocessing Module for RecoMart Pipeline.
Handles cleaning missing/invalid records, encoding categorical features, scaling numerical variables,
and generating Exploratory Data Analysis (EDA) visualizations.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("RecoMart_Preparation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/preparation.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DataPreparer:
    def __init__(self, raw_lake_dir="data_lake/raw", prepared_lake_dir="data_lake/prepared", figures_dir="reports/figures"):
        self.raw_lake_dir = raw_lake_dir
        self.prepared_lake_dir = prepared_lake_dir
        self.figures_dir = figures_dir
        os.makedirs(self.prepared_lake_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def get_latest_file(self, source_type, file_extension="csv"):
        pattern = os.path.join(self.raw_lake_dir, source_type, "**", f"*.{file_extension}")
        files = glob.glob(pattern, recursive=True)
        if not files:
            raise FileNotFoundError(f"No raw files found for source type '{source_type}'")
        return max(files, key=os.path.getmtime)

    def clean_transactions(self):
        tx_file = self.get_latest_file("transactions", "csv")
        logger.info(f"Preparing transactions from '{tx_file}'")
        df = pd.read_csv(tx_file)
        
        # 1. Drop rows missing critical IDs
        id_cols = ["user_id", "item_id"]
        if "transaction_id" in df.columns:
            id_cols.append("transaction_id")
        df = df.dropna(subset=id_cols)
        df['user_id'] = df['user_id'].astype(str)
        df['item_id'] = df['item_id'].astype(str)

        # 2. Filter valid prices and quantities if present
        if 'purchase_price' in df.columns:
            df = df[df['purchase_price'] > 0]
        if 'quantity' in df.columns:
            df = df[df['quantity'] >= 1]
            df['quantity'] = df['quantity'].astype(int)

        # 3. Filter / Clamp ratings to [1.0, 5.0] if present
        if 'rating' in df.columns:
            df = df[(df['rating'] >= 1.0) & (df['rating'] <= 5.0)]
        else:
            df['rating'] = 3.0

        # 4. Parse and standardize purchase_time / timestamp
        time_col = 'purchase_time' if 'purchase_time' in df.columns else 'timestamp'
        df['purchase_time'] = pd.to_datetime(df[time_col], errors='coerce')
        df['timestamp'] = df['purchase_time']
        df = df.dropna(subset=['purchase_time'])
        df['timestamp_epoch'] = df['purchase_time'].astype('int64') // 10**9

        # 5. Calculate transaction total purchase amount
        if 'purchase_price' in df.columns and 'quantity' in df.columns:
            df['total_amount'] = round(df['purchase_price'] * df['quantity'], 2)

        # 6. Encode categorical attributes
        interaction_type_col = 'interaction_type' if 'interaction_type' in df.columns else None
        if interaction_type_col:
            interaction_mapping = {"view": 1.0, "add_to_cart": 2.0, "purchase": 3.0}
            df['interaction_weight'] = df[interaction_type_col].map(interaction_mapping).fillna(1.0)
        else:
            df['interaction_weight'] = 3.0

        if 'device' in df.columns:
            df['device_code'] = df['device'].astype('category').cat.codes
        else:
            df['device_code'] = 0

        # Composite interaction score
        df['implicit_rating'] = df['rating'] * df['interaction_weight']

        output_file = os.path.join(self.prepared_lake_dir, "transactions_prepared.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved prepared transactions ({len(df)} rows) to '{output_file}'")
        return df

    def clean_items(self):
        meta_file = self.get_latest_file("metadata", "json")
        logger.info(f"Preparing item metadata from '{meta_file}'")
        df = pd.read_json(meta_file)
        
        df = df.dropna(subset=["item_id", "price"])
        df['item_id'] = df['item_id'].astype(str)

        # Numerical Normalization (Min-Max Scaling)
        price_min, price_max = df['price'].min(), df['price'].max()
        df['price_normalized'] = (df['price'] - price_min) / (price_max - price_min + 1e-6)

        pop_min, pop_max = df['popularity_score'].min(), df['popularity_score'].max()
        df['popularity_normalized'] = (df['popularity_score'] - pop_min) / (pop_max - pop_min + 1e-6)

        # Categorical Encoding
        df['category_code'] = df['category'].astype('category').cat.codes
        df['brand_code'] = df['brand'].astype('category').cat.codes

        output_file = os.path.join(self.prepared_lake_dir, "items_prepared.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved prepared items ({len(df)} rows) to '{output_file}'")
        return df

    def clean_users(self):
        user_file = self.get_latest_file("users", "csv")
        logger.info(f"Preparing user demographics from '{user_file}'")
        df = pd.read_csv(user_file)
        
        df = df.dropna(subset=["user_id"])
        df['user_id'] = df['user_id'].astype(str)

        # Encoding membership tier
        tier_map = {"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4}
        df['tier_code'] = df['membership_tier'].map(tier_map).fillna(1)
        df['gender_code'] = df['gender'].astype('category').cat.codes
        
        # Scaling age
        df['age_normalized'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min() + 1e-6)

        output_file = os.path.join(self.prepared_lake_dir, "users_prepared.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved prepared users ({len(df)} rows) to '{output_file}'")
        return df

    def generate_eda_visualizations(self, tx_df, items_df):
        logger.info("Generating EDA visualization plots...")
        
        # 1. Rating & Interaction Distribution
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(tx_df['rating'], bins=9, kde=True, ax=ax[0], color='#2563EB')
        ax[0].set_title("User Rating Distribution")
        ax[0].set_xlabel("Rating (1.0 - 5.0)")

        sns.countplot(data=tx_df, x='interaction_type', ax=ax[1], palette='Blues_r')
        ax[1].set_title("Interaction Type Breakdown")
        ax[1].set_xlabel("Type")
        plt.tight_layout()
        dist_plot = os.path.join(self.figures_dir, "interaction_distribution.png")
        plt.savefig(dist_plot, dpi=200)
        plt.close()

        # 2. Top Item Popularity Plot
        top_items = tx_df['item_id'].value_counts().head(15).reset_index()
        top_items.columns = ['item_id', 'count']
        top_items = top_items.merge(items_df[['item_id', 'product_name']], on='item_id', how='left')
        
        plt.figure(figsize=(10, 5))
        sns.barplot(data=top_items, x='count', y='product_name', palette='crest')
        plt.title("Top 15 Most Frequently Interacted Items")
        plt.xlabel("Interaction Count")
        plt.ylabel("Product Name")
        plt.tight_layout()
        pop_plot = os.path.join(self.figures_dir, "item_popularity.png")
        plt.savefig(pop_plot, dpi=200)
        plt.close()

        # 3. User-Item Interaction Matrix Sparsity Heatmap (Sample)
        pivot_sample = tx_df.pivot_table(index='user_id', columns='item_id', values='rating', aggfunc='mean')
        sample_users = pivot_sample.index[:30]
        sample_items = pivot_sample.columns[:30]
        matrix_sample = pivot_sample.loc[sample_users, sample_items].fillna(0)

        num_users = tx_df['user_id'].nunique()
        num_items = tx_df['item_id'].nunique()
        total_possible = num_users * num_items
        actual_interactions = len(tx_df[['user_id', 'item_id']].drop_duplicates())
        sparsity = (1.0 - (actual_interactions / total_possible)) * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_sample > 0, cmap="YlGnBu", cbar=False, linewidths=0.5)
        plt.title(f"User-Item Interaction Sparsity Pattern (Matrix Sparsity: {sparsity:.2f}%)")
        plt.xlabel("Item Index Sample")
        plt.ylabel("User Index Sample")
        plt.tight_layout()
        sparsity_plot = os.path.join(self.figures_dir, "sparsity_heatmap.png")
        plt.savefig(sparsity_plot, dpi=200)
        plt.close()

        logger.info(f"EDA plots generated in '{self.figures_dir}' (Sparsity: {sparsity:.2f}%)")
        return {
            "num_users": num_users,
            "num_items": num_items,
            "total_interactions": len(tx_df),
            "sparsity_pct": round(sparsity, 2)
        }


def run_preparation():
    preparer = DataPreparer()
    tx_df = preparer.clean_transactions()
    items_df = preparer.clean_items()
    users_df = preparer.clean_users()
    eda_stats = preparer.generate_eda_visualizations(tx_df, items_df)
    return eda_stats

if __name__ == "__main__":
    run_preparation()

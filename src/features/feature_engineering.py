"""
Feature Engineering & Transformation Module for RecoMart Pipeline.
Computes user activity aggregates, item rating statistics, recency metrics, and item co-occurrence similarities.
Stores features in a SQLite feature warehouse.
"""

import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger("RecoMart_FeatureEngineering")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/feature_engineering.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class FeatureEngine:
    def __init__(self, prepared_dir="data_lake/prepared", db_path="data_lake/feature_store.db", schema_path="src/features/schema.sql"):
        self.prepared_dir = prepared_dir
        self.db_path = db_path
        self.schema_path = schema_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()

    def init_database(self):
        logger.info(f"Initializing SQLite database at '{self.db_path}'")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if os.path.exists(self.schema_path):
            with open(self.schema_path, "r") as f:
                schema_sql = f.read()
            cursor.executescript(schema_sql)
        conn.commit()
        conn.close()

    def compute_user_features(self, tx_df, users_df):
        logger.info("Computing user-level aggregation features...")
        
        # User rating aggregations
        user_aggs = tx_df.groupby("user_id").agg(
            user_interaction_count=("item_id", "count"),
            user_avg_rating=("rating", "mean"),
            user_rating_std=("rating", "std"),
            last_active_timestamp=("timestamp", "max")
        ).reset_index()
        user_aggs["user_rating_std"] = user_aggs["user_rating_std"].fillna(0.0)

        # Favorite category per user
        tx_merged = tx_df.merge(users_df[["user_id", "tier_code"]], on="user_id", how="left")
        
        # Merge with demographic baseline
        user_features = users_df[["user_id", "tier_code"]].merge(user_aggs, on="user_id", how="left")
        user_features["user_interaction_count"] = user_features["user_interaction_count"].fillna(0).astype(int)
        user_features["user_avg_rating"] = user_features["user_avg_rating"].fillna(0.0)
        user_features["user_rating_std"] = user_features["user_rating_std"].fillna(0.0)
        user_features["user_favorite_category"] = "General"
        user_features["feature_version"] = "v1.0"

        return user_features

    def compute_item_features(self, tx_df, items_df):
        logger.info("Computing item-level aggregation and sentiment features...")
        
        item_aggs = tx_df.groupby("item_id").agg(
            item_interaction_count=("user_id", "count"),
            item_avg_rating=("rating", "mean"),
            item_rating_std=("rating", "std")
        ).reset_index()
        item_aggs["item_rating_std"] = item_aggs["item_rating_std"].fillna(0.0)

        item_features = items_df[["item_id", "price_normalized", "sentiment_score", "popularity_normalized", "category_code"]].merge(
            item_aggs, on="item_id", how="left"
        )
        item_features["item_interaction_count"] = item_features["item_interaction_count"].fillna(0).astype(int)
        item_features["item_avg_rating"] = item_features["item_avg_rating"].fillna(0.0)
        item_features["item_rating_std"] = item_features["item_rating_std"].fillna(0.0)
        item_features["feature_version"] = "v1.0"

        return item_features

    def compute_cooccurrence_features(self, tx_df):
        logger.info("Computing item co-occurrence and Jaccard similarity matrix...")
        
        # Filter purchases / high rating interactions for co-occurrence
        positive_tx = tx_df[tx_df['rating'] >= 3.0]
        user_items = positive_tx.groupby('user_id')['item_id'].unique()

        cooccur = {}
        item_user_counts = positive_tx.groupby('item_id')['user_id'].nunique().to_dict()

        for items in user_items:
            items_list = sorted(list(set(items)))
            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    pair = (items_list[i], items_list[j])
                    cooccur[pair] = cooccur.get(pair, 0) + 1

        cooccur_rows = []
        for (item_a, item_b), count in cooccur.items():
            users_a = item_user_counts.get(item_a, 1)
            users_b = item_user_counts.get(item_b, 1)
            jaccard = count / (users_a + users_b - count + 1e-6)
            
            cooccur_rows.append({
                "item_id_a": item_a,
                "item_id_b": item_b,
                "cooccurrence_count": count,
                "jaccard_similarity": round(float(jaccard), 4),
                "feature_version": "v1.0"
            })
            # Symmetric pair
            cooccur_rows.append({
                "item_id_a": item_b,
                "item_id_b": item_a,
                "cooccurrence_count": count,
                "jaccard_similarity": round(float(jaccard), 4),
                "feature_version": "v1.0"
            })

        cooccur_df = pd.DataFrame(cooccur_rows) if cooccur_rows else pd.DataFrame(columns=["item_id_a", "item_id_b", "cooccurrence_count", "jaccard_similarity", "feature_version"])
        return cooccur_df

    def save_features_to_db(self, user_df, item_df, cooccur_df):
        logger.info(f"Persisting feature tables to SQLite database at '{self.db_path}'")
        conn = sqlite3.connect(self.db_path)
        
        user_df.to_sql("user_features", conn, if_exists="replace", index=False)
        item_df.to_sql("item_features", conn, if_exists="replace", index=False)
        cooccur_df.to_sql("item_cooccurrence", conn, if_exists="replace", index=False)
        
        # Save Metadata Catalog
        metadata = [
            ("user_features", "user_id", ",".join(user_df.columns), "user_features", datetime.now().isoformat(), "v1.0"),
            ("item_features", "item_id", ",".join(item_df.columns), "item_features", datetime.now().isoformat(), "v1.0"),
            ("item_cooccurrence", "item_id_a,item_id_b", ",".join(cooccur_df.columns), "item_cooccurrence", datetime.now().isoformat(), "v1.0")
        ]
        meta_df = pd.DataFrame(metadata, columns=["feature_view_name", "entity_id", "features_list", "source_table", "created_at", "version"])
        meta_df.to_sql("feature_store_metadata", conn, if_exists="replace", index=False)

        conn.commit()
        conn.close()
        logger.info("Successfully updated feature warehouse and metadata registry.")


def run_feature_engineering():
    tx_df = pd.read_csv("data_lake/prepared/transactions_prepared.csv")
    items_df = pd.read_csv("data_lake/prepared/items_prepared.csv")
    users_df = pd.read_csv("data_lake/prepared/users_prepared.csv")

    fe = FeatureEngine()
    user_feats = fe.compute_user_features(tx_df, users_df)
    item_feats = fe.compute_item_features(tx_df, items_df)
    cooccur_feats = fe.compute_cooccurrence_features(tx_df)

    fe.save_features_to_db(user_feats, item_feats, cooccur_feats)
    return user_feats, item_feats, cooccur_feats

if __name__ == "__main__":
    run_feature_engineering()

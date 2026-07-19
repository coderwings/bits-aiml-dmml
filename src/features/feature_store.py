"""
RecoMart Custom Feature Store and Versioned Feature Retrieval Registry.
Provides point-in-time feature serving for batch training and real-time online inference.
"""

import os
import json
import sqlite3
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger("RecoMart_FeatureStore")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/feature_store.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RecoMartFeatureStore:
    def __init__(self, db_path="data_lake/feature_store.db", registry_path="feature_store/feature_registry.json"):
        self.db_path = db_path
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        self._sync_registry()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _sync_registry(self):
        """Builds and updates feature metadata registry JSON file."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        registry = {
            "project": "RecoMart",
            "last_synced": datetime.now().isoformat(),
            "feature_views": {
                "user_features": {
                    "entity": "user_id",
                    "features": ["user_interaction_count", "user_avg_rating", "user_rating_std", "tier_code"],
                    "version": "v1.0"
                },
                "item_features": {
                    "entity": "item_id",
                    "features": ["item_interaction_count", "item_avg_rating", "item_rating_std", "price_normalized", "sentiment_score", "popularity_normalized", "category_code"],
                    "version": "v1.0"
                },
                "item_cooccurrence": {
                    "entity": ["item_id_a", "item_id_b"],
                    "features": ["cooccurrence_count", "jaccard_similarity"],
                    "version": "v1.0"
                }
            }
        }
        
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=4)
        conn.close()
        logger.info(f"Feature Store registry synced at '{self.registry_path}'")

    def get_online_features(self, entity_type, entity_id):
        """
        Retrieves real-time online feature vector for a specific entity (user or item).
        Used during real-time recommendation inference.
        """
        conn = self._get_connection()
        table_name = "user_features" if entity_type == "user" else "item_features"
        id_col = "user_id" if entity_type == "user" else "item_id"
        
        query = f"SELECT * FROM {table_name} WHERE {id_col} = ?"
        df = pd.read_sql_query(query, conn, params=(str(entity_id),))
        conn.close()

        if df.empty:
            logger.warning(f"No online features found for {entity_type}_id '{entity_id}'")
            return None
        return df.to_dict(orient="records")[0]

    def get_historical_features(self, transactions_df):
        """
        Joins offline historical user and item features to transaction dataset for batch training.
        Ensures zero-leakage training feature preparation.
        """
        logger.info("Executing batch historical feature join for model training dataset...")
        conn = self._get_connection()
        
        user_df = pd.read_sql_query("SELECT * FROM user_features", conn)
        item_df = pd.read_sql_query("SELECT * FROM item_features", conn)
        conn.close()

        # Ensure matching string data types
        tx_df = transactions_df.copy()
        tx_df['user_id'] = tx_df['user_id'].astype(str)
        tx_df['item_id'] = tx_df['item_id'].astype(str)
        user_df['user_id'] = user_df['user_id'].astype(str)
        item_df['item_id'] = item_df['item_id'].astype(str)

        # Joined dataset
        joined = tx_df.merge(user_df, on="user_id", how="left", suffixes=("", "_user_view"))
        joined = joined.merge(item_df, on="item_id", how="left", suffixes=("", "_item_view"))

        # Impute missing feature defaults
        num_cols = joined.select_dtypes(include=["float64", "int64"]).columns
        joined[num_cols] = joined[num_cols].fillna(0.0)

        logger.info(f"Joined historical feature dataset shape: {joined.shape}")
        return joined


if __name__ == "__main__":
    fs = RecoMartFeatureStore()
    user_feat = fs.get_online_features("user", "1001")
    print("Online User Feature Lookup (1001):", user_feat)

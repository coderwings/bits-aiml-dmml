"""
RecoMart Real-Time Recommendation Inference Service.
Provides low-latency recommendation endpoint lookups utilizing trained model artifacts and feature store views.
"""

import os
import pickle
import logging
import pandas as pd
from src.features.feature_store import RecoMartFeatureStore

logger = logging.getLogger("RecoMart_Inference")
logger.setLevel(logging.INFO)


class RecommendationInferenceService:
    def __init__(self, svd_path="models/svd_model.pkl", content_path="models/content_model.pkl", prepared_dir="data_lake/prepared"):
        self.svd_path = svd_path
        self.content_path = content_path
        self.prepared_dir = prepared_dir
        self.feature_store = RecoMartFeatureStore()
        
        self.svd_model = None
        self.content_model = None
        self.items_df = None
        self.tx_df = None
        self._load_resources()

    def _load_resources(self):
        if os.path.exists(self.svd_path):
            with open(self.svd_path, "rb") as f:
                self.svd_model = pickle.load(f)
        if os.path.exists(self.content_path):
            with open(self.content_path, "rb") as f:
                self.content_model = pickle.load(f)
        
        items_path = os.path.join(self.prepared_dir, "items_prepared.csv")
        tx_path = os.path.join(self.prepared_dir, "transactions_prepared.csv")
        
        if os.path.exists(items_path):
            self.items_df = pd.read_csv(items_path)
            self.items_df['item_id'] = self.items_df['item_id'].astype(str)
        if os.path.exists(tx_path):
            self.tx_df = pd.read_csv(tx_path)

    def get_recommendations(self, user_id, top_k=5):
        """
        Generates top-K personalized product recommendations for a user.
        Combines SVD Collaborative score, item popularity, and online feature store metadata.
        """
        if self.svd_model is None:
            raise RuntimeError("Model artifacts not loaded. Train models first.")

        u_str = str(user_id)
        # Fetch real-time online features for user
        user_online_feats = self.feature_store.get_online_features("user", u_str)
        
        recs = self.svd_model.recommend_top_k(u_str, k=top_k, tx_df=self.tx_df)
        
        results = []
        for item_id, score in recs:
            item_info = {"item_id": item_id, "predicted_rating": score}
            if self.items_df is not None:
                item_row = self.items_df[self.items_df['item_id'] == str(item_id)]
                if not item_row.empty:
                    item_info["product_name"] = item_row.iloc[0]["product_name"]
                    item_info["category"] = item_row.iloc[0]["category"]
                    item_info["price"] = float(item_row.iloc[0]["price"])
            results.append(item_info)

        return {
            "user_id": u_str,
            "user_tier": user_online_feats.get("tier_code", 1) if user_online_feats else 1,
            "top_k_recommendations": results
        }

if __name__ == "__main__":
    service = RecommendationInferenceService()
    recs = service.get_recommendations("1001", top_k=5)
    print("Inference Recommendation Response:")
    print(pd.DataFrame(recs["top_k_recommendations"]))

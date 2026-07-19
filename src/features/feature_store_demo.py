"""
Feature Store Demonstration Script for RecoMart Pipeline.
Demonstrates batch feature retrieval for model training and online feature serving for real-time inference.
"""

import pandas as pd
from src.features.feature_store import RecoMartFeatureStore

def run_feature_store_demo():
    print("==================================================")
    print("       RecoMart Feature Store Demonstration       ")
    print("==================================================")
    
    fs = RecoMartFeatureStore()
    
    # 1. Online Real-Time Inference Feature Lookup
    print("\n--- [Demo 1] Real-Time Online Feature Retrieval ---")
    sample_user_id = "1001"
    sample_item_id = "501"
    
    user_vector = fs.get_online_features("user", sample_user_id)
    item_vector = fs.get_online_features("item", sample_item_id)
    
    print(f"User '{sample_user_id}' Feature Vector:")
    print(user_vector)
    
    print(f"\nItem '{sample_item_id}' Feature Vector:")
    print(item_vector)

    # 2. Offline Batch Training Feature Join
    print("\n--- [Demo 2] Offline Batch Training Feature Retrieval ---")
    sample_tx = pd.DataFrame([
        {"user_id": "1001", "item_id": "501", "rating": 4.5, "timestamp": "2024-03-01 12:00:00"},
        {"user_id": "1002", "item_id": "502", "rating": 3.0, "timestamp": "2024-03-01 12:30:00"}
    ])
    
    joined_training_df = fs.get_historical_features(sample_tx)
    print(f"\nBatch Training Dataset Shape: {joined_training_df.shape}")
    print("\nSample Training Row Columns:")
    print(list(joined_training_df.columns))
    print("==================================================")

if __name__ == "__main__":
    run_feature_store_demo()

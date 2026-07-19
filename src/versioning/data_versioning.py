"""
Data Versioning and Lineage Tracking Module for RecoMart Pipeline.
Computes dataset checksums, tracks data lineage from raw lake to feature store,
and writes DVC-compatible version manifests.
"""

import os
import glob
import json
import hashlib
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger("RecoMart_Versioning")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/versioning.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def compute_md5(file_path):
    """Computes MD5 hash checksum for a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class DataVersionManager:
    def __init__(self, manifest_dir="data_lake/manifests"):
        self.manifest_dir = manifest_dir
        os.makedirs(self.manifest_dir, exist_ok=True)

    def version_dataset(self, file_path, dataset_name, stage_name, transform_applied="None"):
        if not os.path.exists(file_path):
            logger.warning(f"File '{file_path}' does not exist for versioning.")
            return None

        file_size = os.path.getsize(file_path)
        md5_hash = compute_md5(file_path)
        
        # Read row count if tabular
        row_count = None
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                row_count = len(df)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
                row_count = len(df)
        except Exception:
            row_count = -1

        version_info = {
            "dataset_name": dataset_name,
            "stage": stage_name,
            "file_path": file_path,
            "md5_checksum": md5_hash,
            "size_bytes": file_size,
            "row_count": row_count,
            "transform_applied": transform_applied,
            "timestamp": datetime.now().isoformat(),
            "version_tag": f"v1.{int(datetime.now().timestamp())}"
        }

        # Write manifest file (.dvc style metadata)
        manifest_filename = f"{dataset_name}_{stage_name}_manifest.json"
        manifest_path = os.path.join(self.manifest_dir, manifest_filename)
        with open(manifest_path, "w") as f:
            json.dump(version_info, f, indent=4)

        logger.info(f"Versioned dataset '{dataset_name}' at stage '{stage_name}' (MD5: {md5_hash[:8]}...)")
        return version_info

    def generate_lineage_graph(self):
        """Builds end-to-end data lineage graph from raw ingestion to model training."""
        logger.info("Generating data lineage graph...")
        lineage = {
            "pipeline": "RecoMart Recommendation Data Management Pipeline",
            "last_updated": datetime.now().isoformat(),
            "lineage_nodes": [
                {
                    "id": "raw_transactions",
                    "stage": "Data Ingestion",
                    "source": "Clickstream CSV",
                    "target": "prepared_transactions"
                },
                {
                    "id": "raw_metadata",
                    "stage": "Data Ingestion",
                    "source": "REST API / JSON Catalog",
                    "target": "prepared_items"
                },
                {
                    "id": "prepared_transactions",
                    "stage": "Data Preparation",
                    "transformations": "Null imputation, rating clamping [1-5], datetime parsing, interaction weighting",
                    "target": "user_features & item_cooccurrence"
                },
                {
                    "id": "prepared_items",
                    "stage": "Data Preparation",
                    "transformations": "MinMax price scaling, sentiment normalization, brand categorical encoding",
                    "target": "item_features"
                },
                {
                    "id": "feature_store_db",
                    "stage": "Feature Store & Warehouse",
                    "tables": ["user_features", "item_features", "item_cooccurrence"],
                    "target": "recommendation_model_training"
                },
                {
                    "id": "svd_model_artifact",
                    "stage": "Model Training",
                    "algorithm": "Matrix Factorization SVD + Content-Based Filtering",
                    "target": "inference_api"
                }
            ]
        }
        
        lineage_path = os.path.join(self.manifest_dir, "data_lineage_graph.json")
        with open(lineage_path, "w") as f:
            json.dump(lineage, f, indent=4)
        return lineage


def run_versioning():
    vm = DataVersionManager()
    
    # Version raw files
    raw_tx = glob.glob("data_lake/raw/transactions/**/*.csv", recursive=True)
    if raw_tx:
        vm.version_dataset(raw_tx[0], "transactions", "raw", "Raw Batch CSV Ingestion")

    # Version prepared files
    if os.path.exists("data_lake/prepared/transactions_prepared.csv"):
        vm.version_dataset("data_lake/prepared/transactions_prepared.csv", "transactions", "prepared", "Data Cleaning & Categorical Encoding")
    
    if os.path.exists("data_lake/prepared/items_prepared.csv"):
        vm.version_dataset("data_lake/prepared/items_prepared.csv", "items", "prepared", "MinMax Scaling & Sentiment Normalization")

    # Version feature store DB
    if os.path.exists("data_lake/feature_store.db"):
        vm.version_dataset("data_lake/feature_store.db", "feature_store", "warehouse", "SQLite Aggregation Feature Store")

    lineage = vm.generate_lineage_graph()
    return lineage

if __name__ == "__main__":
    run_versioning()

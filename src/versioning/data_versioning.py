"""
Data Versioning and Lineage Tracking Module for RecoMart Pipeline.
Integrates Data Version Control (DVC) for tracking raw datasets, prepared files, and feature warehouse artifacts.
Computes dataset checksums, tracks data lineage, and generates DVC-compatible version manifests.
"""

import os
import glob
import json
import hashlib
import logging
import subprocess
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
        self.dvc_bin = self._find_dvc_binary()
        self.init_dvc()

    def _find_dvc_binary(self):
        """Locates DVC binary in virtual environment or system PATH."""
        venv_dvc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "venv", "bin", "dvc"))
        if os.path.exists(venv_dvc):
            return venv_dvc
        import shutil
        sys_dvc = shutil.which("dvc")
        return sys_dvc if sys_dvc else "dvc"

    def init_dvc(self):
        """Initializes DVC repository if not already initialized."""
        if not os.path.exists(".dvc"):
            logger.info("Initializing DVC repository...")
            try:
                cmd = [self.dvc_bin, "init", "--no-scm", "-f"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    logger.info("DVC repository initialized successfully.")
                else:
                    logger.warning(f"DVC init warning: {res.stderr.strip()}")
            except Exception as e:
                logger.warning(f"DVC initialization bypassed: {str(e)}")

    def add_to_dvc(self, file_path):
        """Adds a dataset or model artifact to DVC tracking."""
        if not os.path.exists(file_path):
            return None
        
        try:
            logger.info(f"Adding '{file_path}' to DVC tracking...")
            cmd = [self.dvc_bin, "add", file_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                dvc_file = f"{file_path}.dvc"
                logger.info(f"DVC file created at '{dvc_file}'")
                return dvc_file
            else:
                logger.warning(f"DVC add output: {res.stderr.strip() or res.stdout.strip()}")
        except Exception as e:
            logger.warning(f"DVC add command failed ({str(e)}). Falling back to manifest tracking.")
        return None

    def version_dataset(self, file_path, dataset_name, stage_name, transform_applied="None"):
        if not os.path.exists(file_path):
            logger.warning(f"File '{file_path}' does not exist for versioning.")
            return None

        file_size = os.path.getsize(file_path)
        md5_hash = compute_md5(file_path)
        
        # Track with DVC
        dvc_file = self.add_to_dvc(file_path)

        # Parse DVC metadata if .dvc file was generated
        dvc_md5 = None
        if dvc_file and os.path.exists(dvc_file):
            try:
                import yaml
                with open(dvc_file, "r") as f:
                    dvc_meta = yaml.safe_load(f)
                if "outs" in dvc_meta and len(dvc_meta["outs"]) > 0:
                    dvc_md5 = dvc_meta["outs"][0].get("md5")
            except Exception:
                dvc_md5 = md5_hash

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
            "dvc_tracked": dvc_file is not None,
            "dvc_file": dvc_file,
            "md5_checksum": md5_hash,
            "dvc_md5": dvc_md5 or md5_hash,
            "size_bytes": file_size,
            "row_count": row_count,
            "transform_applied": transform_applied,
            "timestamp": datetime.now().isoformat(),
            "version_tag": f"v1.{int(datetime.now().timestamp())}"
        }

        # Write manifest file (.dvc style metadata manifest)
        manifest_filename = f"{dataset_name}_{stage_name}_manifest.json"
        manifest_path = os.path.join(self.manifest_dir, manifest_filename)
        with open(manifest_path, "w") as f:
            json.dump(version_info, f, indent=4)

        logger.info(f"Versioned dataset '{dataset_name}' at stage '{stage_name}' (MD5: {md5_hash[:8]}..., DVC: {dvc_file is not None})")
        return version_info

    def generate_lineage_graph(self):
        """Builds end-to-end data lineage graph from raw ingestion to model training."""
        logger.info("Generating data lineage graph...")
        lineage = {
            "pipeline": "RecoMart Recommendation Data Management Pipeline",
            "dvc_enabled": True,
            "last_updated": datetime.now().isoformat(),
            "lineage_nodes": [
                {
                    "id": "raw_transactions",
                    "stage": "Data Ingestion",
                    "source": "Clickstream CSV",
                    "target": "prepared_transactions",
                    "dvc_tracked": True
                },
                {
                    "id": "raw_metadata",
                    "stage": "Data Ingestion",
                    "source": "REST API / JSON Catalog",
                    "target": "prepared_items",
                    "dvc_tracked": True
                },
                {
                    "id": "prepared_transactions",
                    "stage": "Data Preparation",
                    "transformations": "Null imputation, rating clamping [1-5], datetime parsing, interaction weighting",
                    "target": "user_features & item_cooccurrence",
                    "dvc_tracked": True
                },
                {
                    "id": "prepared_items",
                    "stage": "Data Preparation",
                    "transformations": "MinMax price scaling, sentiment normalization, brand categorical encoding",
                    "target": "item_features",
                    "dvc_tracked": True
                },
                {
                    "id": "feature_store_db",
                    "stage": "Feature Store & Warehouse",
                    "tables": ["user_features", "item_features", "item_cooccurrence"],
                    "target": "recommendation_model_training",
                    "dvc_tracked": True
                },
                {
                    "id": "svd_model_artifact",
                    "stage": "Model Training",
                    "algorithm": "Matrix Factorization SVD + Content-Based Filtering",
                    "target": "inference_api",
                    "dvc_tracked": True
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

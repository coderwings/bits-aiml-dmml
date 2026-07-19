"""
Data Ingestion Module for RecoMart Recommendation System.
Handles automated batch ingestion of user transactions, product metadata, and user demographics.
Implements retry logic, error handling, structured logging, and date-partitioned storage.
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Ensure log directory exists
os.makedirs('logs', exist_ok=True)

# Configure logger
logger = logging.getLogger("RecoMart_Ingestion")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/ingestion.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def get_http_session(retries=3, backoff_factor=0.5):
    """Returns a requests Session configured with automatic HTTP retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_partition_path(base_dir, source_type, dt=None):
    """Generates partitioned directory path: base_dir/source_type/year=YYYY/month=MM/day=DD."""
    if dt is None:
        dt = datetime.now()
    path = os.path.join(
        base_dir,
        source_type,
        f"year={dt.year}",
        f"month={dt.month:02d}",
        f"day={dt.day:02d}"
    )
    os.makedirs(path, exist_ok=True)
    return path


def ingest_transaction_csv(input_path="input_data/transactions.csv", raw_lake_dir="data_lake/raw"):
    """Ingests user transaction logs from CSV into raw partitioned data lake."""
    try:
        logger.info(f"Starting transaction ingestion from '{input_path}'")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Transaction file '{input_path}' not found.")
        
        df = pd.read_csv(input_path)
        dt = datetime.now()
        partition_path = get_partition_path(raw_lake_dir, "transactions", dt)
        timestamp = dt.strftime("%Y%m%d_%H%M%S")
        dest_file = os.path.join(partition_path, f"transactions_{timestamp}.csv")
        
        df.to_csv(dest_file, index=False)
        logger.info(f"Successfully ingested {len(df)} transactions to '{dest_file}'")
        return dest_file
    except Exception as e:
        logger.error(f"Failed transaction CSV ingestion: {str(e)}", exc_info=True)
        raise e


def ingest_user_csv(input_path="input_data/users.csv", raw_lake_dir="data_lake/raw"):
    """Ingests user demographics CSV into raw partitioned data lake."""
    try:
        logger.info(f"Starting user demographics ingestion from '{input_path}'")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Users file '{input_path}' not found.")
        
        df = pd.read_csv(input_path)
        dt = datetime.now()
        partition_path = get_partition_path(raw_lake_dir, "users", dt)
        timestamp = dt.strftime("%Y%m%d_%H%M%S")
        dest_file = os.path.join(partition_path, f"users_{timestamp}.csv")
        
        df.to_csv(dest_file, index=False)
        logger.info(f"Successfully ingested {len(df)} user profiles to '{dest_file}'")
        return dest_file
    except Exception as e:
        logger.error(f"Failed user CSV ingestion: {str(e)}", exc_info=True)
        raise e


def ingest_external_metadata_api(api_url=None, fallback_input_path="input_data/items.json", raw_lake_dir="data_lake/raw"):
    """
    Ingests item metadata from REST API endpoint or fallback local payload.
    Supports HTTP retries and JSON payload validation.
    """
    session = get_http_session()
    try:
        data = None
        if api_url:
            logger.info(f"Attempting ingestion from REST API endpoint: {api_url}")
            try:
                response = session.get(api_url, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.info("REST API fetch successful.")
            except Exception as api_err:
                logger.warning(f"REST API fetch failed ({str(api_err)}). Falling back to catalog source '{fallback_input_path}'")

        if data is None:
            if not os.path.exists(fallback_input_path):
                raise FileNotFoundError(f"Fallback item file '{fallback_input_path}' does not exist.")
            with open(fallback_input_path, "r") as f:
                data = json.load(f)

        df = pd.DataFrame(data)
        dt = datetime.now()
        partition_path = get_partition_path(raw_lake_dir, "metadata", dt)
        timestamp = dt.strftime("%Y%m%d_%H%M%S")
        dest_file = os.path.join(partition_path, f"metadata_{timestamp}.json")
        
        df.to_json(dest_file, orient='records', indent=4)
        logger.info(f"Successfully ingested {len(df)} item metadata records to '{dest_file}'")
        return dest_file
    except Exception as e:
        logger.error(f"Failed item metadata ingestion: {str(e)}", exc_info=True)
        raise e


def run_ingestion():
    """Master trigger function for data ingestion stage."""
    logger.info("=== Starting RecoMart Data Ingestion Stage ===")
    
    # Ensure synthetic inputs exist if not present
    from input_data.generate_synthetic_data import generate_synthetic_data
    if not os.path.exists("input_data/transactions.csv"):
        logger.info("Generating initial synthetic dataset for pipeline run...")
        generate_synthetic_data()

    tx_file = ingest_transaction_csv()
    usr_file = ingest_user_csv()
    meta_file = ingest_external_metadata_api()
    
    logger.info("=== RecoMart Data Ingestion Stage Completed Successfully ===")
    return {
        "transactions_file": tx_file,
        "users_file": usr_file,
        "metadata_file": meta_file
    }

if __name__ == "__main__":
    run_ingestion()

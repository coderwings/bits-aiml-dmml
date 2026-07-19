import os
import logging
import requests
import pandas as pd
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Remove any existing handlers to ensure basicConfig takes effect in notebooks
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure structured logging
logging.basicConfig(
    filename='logs/ingestion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

def ingest_transaction_csv(file_path, output_dir):
    try:
        logging.info(f"Starting file ingestion from historical source: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file {file_path} does not exist.")
        df = pd.read_csv(file_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partition_path = os.path.join(output_dir, f"year={datetime.now().year}/month={datetime.now().month:02d}")
        os.makedirs(partition_path, exist_ok=True)

        dest_file = os.path.join(partition_path, f"transactions_{timestamp}.csv")
        df.to_csv(dest_file, index=False)
        logging.info(f"Successfully stored batch transactions to data_lake : {dest_file}")
    except Exception as e:
        logging.error(f"Failed to ingest transactions CSV: {str(e)}", exc_info=True)
        raise e


def get_http_session(retries=3, backoff_factor=0.5):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    return session

def get_data_for_api():
    """
    Placeholder function to simulate an external API providing product information
    with sentiment or popularity scores.
    """
    return [
        {
            "item_id": "501",
            "product_name": "Laptop Pro",
            "sentiment_score": 0.85,
            "popularity_score": 9.2
        },
        {
            "item_id": "502",
            "product_name": "Wireless Mouse",
            "sentiment_score": 0.70,
            "popularity_score": 7.5
        },
        {
            "item_id": "503",
            "product_name": "Mechanical Keyboard",
            "sentiment_score": 0.92,
            "popularity_score": 9.8
        }
    ]

def ingest_external_metadata_api(api_url, output_dir):
    session = get_http_session()
    try:
        logging.info(f"Querying external REST API endpoints: {api_url}")
        # uncomment when we have API providing sentiment or popularity scores
        #response = session.get(api_url, timeout=10)
        #response.raise_for_status()
        #data = response.json()

        data = get_data_for_api()

        df = pd.DataFrame(data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partition_path = os.path.join(output_dir, f"year={datetime.now().year}/month={datetime.now().month:02d}")
        os.makedirs(partition_path, exist_ok=True)

        dest_file = os.path.join(partition_path, f"metadata_{timestamp}.json")
        df.to_json(dest_file, orient='records', indent=4)
        logging.info(f"Successfully stored API metadata payload: {dest_file}")
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP protocol error occurred during API fetch: {str(http_err)}", exc_info=True)
        raise http_err
    except Exception as e:
        logging.error(f"Unexpected operational error during API ingestion: {str(e)}", exc_info=True)
        raise e

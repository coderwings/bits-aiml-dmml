import os
import pandas as pd
import logging # Import logging module

# Define output directory for ingested data
output_dir = "data_lake/transactions"
os.makedirs(output_dir, exist_ok=True)
logging.info(f"Ensured output directory '{output_dir}' exists.")

# --- Call ingest_transaction_csv function ---
transaction_csv_path = "input_data/transactions.csv"
try:
    logging.info(f"Calling ingest_transaction_csv with file_path='{transaction_csv_path}' and output_dir='{output_dir}'")
    ingest_transaction_csv(transaction_csv_path, output_dir)
except Exception as e:
    logging.error(f"Error calling ingest_transaction_csv: {e}")

# --- Call ingest_external_metadata_api function ---
metadata_api_url = "https://external-api.com/" # Example public API

try:
    logging.info(f"Calling ingest_external_metadata_api with api_url='{metadata_api_url}' and output_dir='{output_dir}'")
    ingest_external_metadata_api(metadata_api_url, output_dir)
except Exception as e:
    logging.error(f"Error calling ingest_external_metadata_api: {e}")

logging.info("Ingestion process initiated for both functions. Check the 'data_lake' directory and 'logs/ingestion.log' for details.")

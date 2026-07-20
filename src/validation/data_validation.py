"""
Data Validation and Quality Profiling Module for RecoMart Pipeline.
Executes schema validation, missingness checks, duplicate detection, and range/format constraints.
Generates Data Quality summary report and PDF deliverable.
"""

import os
import glob
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.reports.pdf_generator import create_pdf_report

logger = logging.getLogger("RecoMart_Validation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/validation.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DataValidator:
    def __init__(self, raw_lake_dir="data_lake/raw"):
        self.raw_lake_dir = raw_lake_dir

    def get_latest_file(self, source_type, file_extension="csv"):
        pattern = os.path.join(self.raw_lake_dir, source_type, "**", f"*.{file_extension}")
        files = glob.glob(pattern, recursive=True)
        if not files:
            raise FileNotFoundError(f"No raw files found for source type '{source_type}' in '{self.raw_lake_dir}'")
        return max(files, key=os.path.getmtime)

    def validate_transactions(self, file_path=None):
        if file_path is None:
            file_path = self.get_latest_file("transactions", "csv")
        
        logger.info(f"Validating transaction dataset: '{file_path}'")
        df = pd.read_csv(file_path)
        initial_count = len(df)
        
        report = {
            "dataset": "transactions",
            "file_path": file_path,
            "total_records": initial_count,
            "checks": {},
            "passed": True
        }

        # 1. Schema Check
        required_cols = {"transaction_id", "user_id", "item_id", "purchase_price", "quantity", "purchase_time"}
        missing_cols = list(required_cols - set(df.columns))
        report["checks"]["schema_check"] = {
            "status": "PASS" if not missing_cols else "FAIL",
            "missing_columns": missing_cols
        }

        # 2. Null Values Check
        null_counts = df[list(required_cols.intersection(set(df.columns)))].isnull().sum().to_dict()
        total_null_rows = df[list(required_cols.intersection(set(df.columns)))].isnull().any(axis=1).sum()
        report["checks"]["null_check"] = {
            "status": "PASS" if total_null_rows == 0 else "WARNING",
            "null_row_count": int(total_null_rows),
            "null_column_breakdown": null_counts
        }

        # 3. Duplicate Records Check
        dup_count = int(df.duplicated(subset=["transaction_id"]).sum()) if "transaction_id" in df.columns else int(df.duplicated().sum())
        report["checks"]["duplicate_check"] = {
            "status": "PASS" if dup_count == 0 else "WARNING",
            "duplicate_count": dup_count
        }

        # 4. Purchase Price and Quantity Range Checks
        invalid_price_count = int((df['purchase_price'] <= 0).sum()) if 'purchase_price' in df.columns else 0
        invalid_qty_count = int((df['quantity'] <= 0).sum()) if 'quantity' in df.columns else 0
        report["checks"]["purchase_value_check"] = {
            "status": "PASS" if (invalid_price_count == 0 and invalid_qty_count == 0) else "FAIL",
            "invalid_price_count": invalid_price_count,
            "invalid_quantity_count": invalid_qty_count
        }

        # 5. Purchase Time Format Check
        def check_date(val):
            try:
                pd.to_datetime(val)
                return True
            except:
                return False
        
        time_col = 'purchase_time' if 'purchase_time' in df.columns else 'timestamp'
        invalid_ts_count = int((~df[time_col].apply(check_date)).sum())
        report["checks"]["timestamp_format_check"] = {
            "status": "PASS" if invalid_ts_count == 0 else "FAIL",
            "invalid_timestamp_count": invalid_ts_count
        }

        if any(c["status"] == "FAIL" for c in report["checks"].values()):
            report["passed"] = False

        logger.info(f"Transactions validation completed. Passed: {report['passed']}")
        return report, df

    def validate_metadata(self, file_path=None):
        if file_path is None:
            file_path = self.get_latest_file("metadata", "json")

        logger.info(f"Validating metadata dataset: '{file_path}'")
        df = pd.read_json(file_path)
        initial_count = len(df)

        report = {
            "dataset": "metadata",
            "file_path": file_path,
            "total_records": initial_count,
            "checks": {},
            "passed": True
        }

        required_cols = {"item_id", "product_name", "category", "price", "sentiment_score", "popularity_score"}
        missing_cols = list(required_cols - set(df.columns))
        report["checks"]["schema_check"] = {
            "status": "PASS" if not missing_cols else "FAIL",
            "missing_columns": missing_cols
        }

        price_invalid = int((df['price'] <= 0).sum())
        report["checks"]["price_range_check"] = {
            "status": "PASS" if price_invalid == 0 else "FAIL",
            "invalid_price_count": price_invalid
        }

        logger.info(f"Metadata validation completed. Passed: {report['passed']}")
        return report, df

    def generate_data_quality_report(self, tx_report, meta_report, output_dir="reports"):
        os.makedirs(output_dir, exist_ok=True)
        report_json_path = os.path.join(output_dir, "data_quality_summary.json")
        summary_payload = {
            "timestamp": datetime.now().isoformat(),
            "transactions": tx_report,
            "metadata": meta_report
        }
        with open(report_json_path, "w") as f:
            json.dump(summary_payload, f, indent=4)

        # Markdown Report
        md_path = os.path.join(output_dir, "data_quality_report.md")
        with open(md_path, "w") as f:
            f.write("# RecoMart Automated Data Quality & Validation Report\n\n")
            f.write(f"**Execution Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 1. Transactions Dataset Profile\n")
            f.write(f"- **File Path:** `{tx_report['file_path']}`\n")
            f.write(f"- **Total Rows:** {tx_report['total_records']}\n")
            f.write(f"- **Overall Status:** {'PASSED' if tx_report['passed'] else 'FLAGGED ANOMALIES'}\n\n")
            f.write("| Quality Check | Status | Details |\n|---|---|---|\n")
            for name, details in tx_report["checks"].items():
                f.write(f"| {name} | {details['status']} | {json.dumps(details)} |\n")

            f.write("\n## 2. Product Metadata Profile\n")
            f.write(f"- **File Path:** `{meta_report['file_path']}`\n")
            f.write(f"- **Total Rows:** {meta_report['total_records']}\n")
            f.write(f"- **Overall Status:** {'PASSED' if meta_report['passed'] else 'FLAGGED ANOMALIES'}\n\n")
            f.write("| Quality Check | Status | Details |\n|---|---|---|\n")
            for name, details in meta_report["checks"].items():
                f.write(f"| {name} | {details['status']} | {json.dumps(details)} |\n")

        # Build detailed validation table rows for PDF
        tx_table_rows = []
        for k, v in tx_report["checks"].items():
            count_val = (
                v.get("invalid_timestamp_count") or
                v.get("null_row_count") or
                v.get("duplicate_count") or
                v.get("invalid_price_count") or
                v.get("invalid_rating_count") or
                0
            )
            tx_table_rows.append([k, v["status"], str(count_val)])

        meta_table_rows = []
        for k, v in meta_report["checks"].items():
            count_val = v.get("invalid_price_count") or 0
            meta_table_rows.append([k, v["status"], str(count_val)])

        # PDF Report Generation via ReportLab
        pdf_sections = [
            {
                "heading": "1. Data Validation Executive Summary",
                "content": f"Automated quality checks evaluated raw datasets in the local data lake on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Below is the summary of schema adherence, missing values, duplicates, and range constraints."
            },
            {
                "heading": "2. Transactions Dataset Validation Matrix",
                "table_data": [["Check Name", "Status", "Anomaly Count / Details"]] + tx_table_rows
            },
            {
                "heading": "3. Metadata Dataset Validation Matrix",
                "table_data": [["Check Name", "Status", "Anomaly Count / Details"]] + meta_table_rows
            }
        ]
        create_pdf_report("data_quality_report.pdf", "RecoMart Data Quality & Profiling Report", "Data Validation & Anomaly Detection Results", pdf_sections, output_dir)
        logger.info(f"Generated Data Quality reports in '{output_dir}'")


def run_validation():
    validator = DataValidator()
    tx_report, _ = validator.validate_transactions()
    meta_report, _ = validator.validate_metadata()
    validator.generate_data_quality_report(tx_report, meta_report)
    return tx_report, meta_report


if __name__ == "__main__":
    run_validation()

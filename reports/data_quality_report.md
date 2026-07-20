# RecoMart Automated Data Quality & Validation Report

**Execution Timestamp:** 2026-07-20 22:28:35

## 1. Transactions Dataset Profile
- **File Path:** `data_lake/raw/transactions/year=2026/month=07/day=20/transactions_20260720_222834.csv`
- **Total Rows:** 2000
- **Overall Status:** FLAGGED ANOMALIES

| Quality Check | Status | Details |
|---|---|---|
| schema_check | PASS | {"status": "PASS", "missing_columns": []} |
| null_check | WARNING | {"status": "WARNING", "null_row_count": 5, "null_column_breakdown": {"purchase_price": 0, "purchase_time": 0, "item_id": 0, "quantity": 0, "user_id": 5, "transaction_id": 0}} |
| duplicate_check | PASS | {"status": "PASS", "duplicate_count": 0} |
| purchase_value_check | PASS | {"status": "PASS", "invalid_price_count": 0, "invalid_quantity_count": 0} |
| timestamp_format_check | FAIL | {"status": "FAIL", "invalid_timestamp_count": 5} |

## 2. Product Metadata Profile
- **File Path:** `data_lake/raw/metadata/year=2026/month=07/day=20/metadata_20260720_222834.json`
- **Total Rows:** 50
- **Overall Status:** PASSED

| Quality Check | Status | Details |
|---|---|---|
| schema_check | PASS | {"status": "PASS", "missing_columns": []} |
| price_range_check | PASS | {"status": "PASS", "invalid_price_count": 0} |

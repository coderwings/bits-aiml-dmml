# RecoMart Data Versioning & Lineage Documentation (DVC Integration)

## 1. Data Version Control (DVC) Strategy
The RecoMart Data Management Pipeline uses **DVC (Data Version Control)** to version-control raw datasets, cleaned datasets, and model artifacts without storing large binary files in Git.

### DVC Tracking Workflow
1. **Initialize DVC Repository:**
   ```bash
   dvc init --no-scm
   ```

2. **Track Raw Data & Prepared Artifacts:**
   Datasets are tracked using DVC:
   ```bash
   dvc add data_lake/prepared/transactions_prepared.csv
   dvc add data_lake/prepared/items_prepared.csv
   dvc add data_lake/feature_store.db
   ```
   This generates `.dvc` tracking files (e.g. `transactions_prepared.csv.dvc`, `feature_store.db.dvc`) containing immutable MD5 data content hashes.

3. **Git Integration:**
   Git tracks lightweight `.dvc` files and manifest JSON files, while actual data payloads remain in the local DVC cache (`.dvc/cache`).

---

## 2. Version Manifest Metadata
Every stage writes a DVC-compatible version manifest to `data_lake/manifests/`:

```json
{
    "dataset_name": "transactions",
    "stage": "prepared",
    "file_path": "data_lake/prepared/transactions_prepared.csv",
    "dvc_tracked": true,
    "dvc_file": "data_lake/prepared/transactions_prepared.csv.dvc",
    "md5_checksum": "a754afd5...",
    "dvc_md5": "a754afd5...",
    "row_count": 285,
    "transform_applied": "Data Cleaning & Categorical Encoding",
    "timestamp": "2026-07-20T22:17:15.000000",
    "version_tag": "v1.1768853835"
}
```

---

## 3. End-to-End Lineage Flow
```
[ Clickstream CSV ]  ---> [ Raw Partition Data Lake ] ---> [ Cleaning & Validation ]
  (DVC Tracked)               (DVC Tracked)                      (DVC Tracked)
                                                                       |
[ Product API JSON ] ---> [ Raw Partition Data Lake ] -----------------+
  (DVC Tracked)               (DVC Tracked)                            |
                                                                       v
                                                            [ Prepared Datasets ]
                                                              (DVC Tracked)
                                                                       |
                                                                       v
                                                            [ Feature Store DB ]
                                                              (DVC Tracked)
                                                                       |
                                                                       v
                                                            [ Model Training (SVD) ]
                                                              (DVC Tracked)
                                                                       |
                                                                       v
                                                            [ Inference Service ]
```

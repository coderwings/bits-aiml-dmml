# RecoMart Data Versioning & Lineage Documentation

## Data Versioning Strategy
Every dataset artifact across all stages of the pipeline is checksummed using MD5 hashes and tracked via version manifests in `data_lake/manifests/`.

### Metadata Manifest Attributes
- `dataset_name`: Name of the dataset entity (e.g. `transactions`, `items`).
- `stage`: Pipeline stage (`raw`, `prepared`, `warehouse`).
- `md5_checksum`: Unique MD5 cryptographic hash of the data payload.
- `row_count`: Number of validated data rows.
- `transform_applied`: Applied data preprocessing or cleaning logic.
- `version_tag`: Immutable release version timestamp.

## End-to-End Lineage Flow
```
[ Clickstream CSV ]  ---> [ Raw Partition Data Lake ] ---> [ Cleaning & Validation ]
                                                                   |
[ Product API JSON ] ---> [ Raw Partition Data Lake ] -------------+
                                                                   |
                                                                   v
                                                        [ Prepared Datasets ]
                                                                   |
                                                                   v
                                                        [ Feature Store DB ]
                                                                   |
                                                                   v
                                                        [ Model Training (SVD) ]
                                                                   |
                                                                   v
                                                        [ Inference Service ]
```

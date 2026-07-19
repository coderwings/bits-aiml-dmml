# RecoMart Raw Data Storage Documentation

## Storage Architecture
Raw ingested data is stored in a structured local data lake hierarchy mimicking Cloud Blob Storage (AWS S3 bucket structure).

```
data_lake/
├── raw/
│   ├── transactions/
│   │   └── year=2026/
│   │       └── month=07/
│   │           └── day=19/
│   │               └── transactions_20260719_203452.csv
│   ├── metadata/
│   │   └── year=2026/
│   │       └── month=07/
│   │           └── day=19/
│   │               └── metadata_20260719_203452.json
│   └── users/
│       └── year=2026/
│           └── month=07/
│               └── day=19/
│                   └── users_20260719_203452.csv
```

## Partitioning Strategy
- **Partition Keys:** `source`, `year=YYYY`, `month=MM`, `day=DD`
- **File Format:** CSV for clickstream and transactions, JSON for REST API product metadata payloads.
- **Retention & Immutability:** Raw data lake partition directories are append-only to preserve raw audit trails.

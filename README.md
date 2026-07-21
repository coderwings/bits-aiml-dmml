# RecoMart End-to-End Recommendation Data Management & ML Pipeline

A scalable, production-ready data management pipeline for **RecoMart** (e-commerce platform). This repository implements data ingestion, quality validation, preparation, feature engineering, feature store serving, data versioning, model training (SVD Collaborative + Content-Based Hybrid), metric evaluation, and pipeline orchestration.

---

## Architecture Overview

```
1. Multi-Source Data Ingestion (Transaction CSVs + Product REST API)
2. Partitioned Raw Data Lake Storage (data_lake/raw/year/month/day)
3. Data Profiling & Quality Validation  (Schema, Nulls, Duplicates, Range) 
4. Data Cleaning & Preprocessing (Encoding, Scaling, EDA Figures)
5. Feature Engineering & Feature Store DB (Offline Join + Online Point Lookup) 
6. SVD Collaborative + Content Model Train (Precision@K, Recall@K, NDCG@K)
7. Deployable Inference API Service
```

---

## Repository Directory Structure

```
bits-aiml-dmml/
├── README.md                           # Project documentation
├── requirements.txt                    # Dependencies manifest
├── data_pipeline_main.py               # Master Orchestration CLI entrypoint
├── input_data/                         # Input raw data generators & sources
│   ├── generate_synthetic_data.py      # Synthetic dataset generator
│   ├── transactions.csv                # User clickstream & transaction logs
│   ├── users.csv                       # User demographic profiles
│   └── items.json                      # Product metadata REST payload
├── data_lake/                          # Data Lake Storage Root
│   ├── raw/                            # Partitioned raw lake storage
│   ├── prepared/                       # Preprocessed & cleaned datasets
│   ├── manifests/                      # DVC dataset version manifests & lineage
│   └── feature_store.db                # SQLite Relational Feature Warehouse
├── feature_store/                      # Feature Store Registry
│   └── feature_registry.json           # Feature metadata catalog
├── models/                             # Model Artifact Registry
│   ├── svd_model.pkl                   # Matrix Factorization SVD model
│   ├── content_model.pkl               # Content-Based Cosine Similarity model
│   └── model_registry.json             # Model hyperparameter & metric logs
├── reports/                            # Generated PDF & Markdown Deliverables
│   ├── data_quality_report.pdf / .md
│   ├── model_performance_report.pdf / .md
│   └── figures/                        # EDA Visualizations (Interaction, Popularity, Sparsity)
├── docs/                               # System Documentation
│   ├── raw_data_storage.md
│   ├── feature_summary.md
│   ├── feature_store.md
│   └── data_versioning_lineage.md
├── notebooks/                          # Jupyter Notebooks
│   └── eda_and_preparation.ipynb
├── src/                                # Core Modular Source Code
│   ├── ingestion/                      # Multi-source batch/API ingestion
│   ├── validation/                     # Automated data quality checks
│   ├── preparation/                    # Data cleaning, scaling, and EDA
│   ├── features/                       # Feature engineering & Feature Store
│   ├── versioning/                     # Checksumming & lineage manager
│   ├── model/                          # Recommenders, training, & inference
│   ├── reports/                        # PDF report generator
│   └── orchestration/                  # Pipeline orchestrator & Airflow DAGs
└── tests/                              # Pytest test suite
    └── test_pipeline.py
```

---

## Installation & Setup

1. **Clone the Repository & Navigate to Folder:**
   ```bash
   cd bits-aiml-dmml
   ```

2. **Activate Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Pipeline

### Execute Complete End-to-End Orchestrated Pipeline
Runs all 10 stages sequentially, outputs progress to console, writes logs to `logs/`, and generates reports in `reports/`:

```bash
python3 data_pipeline_main.py
```

---

## Testing

Run automated unit and integration tests using `pytest`:

```bash
pytest tests/test_pipeline.py -v
```

---

## Individual Stage Execution

You can also run individual modules independently:

- **Generate Synthetic Raw Data:**
  ```bash
  python3 input_data/generate_synthetic_data.py
  ```

- **Data Ingestion:**
  ```bash
  python3 -m src.ingestion.data_ingestion
  ```

- **Data Validation & Profiling:**
  ```bash
  python3 -m src.validation.data_validation
  ```

- **Data Preparation & EDA Plot Generation:**
  ```bash
  python3 -m src.preparation.data_preparation
  ```

- **Feature Engineering & Feature Store Sync:**
  ```bash
  python3 -m src.features.feature_engineering
  python3 -m src.features.feature_store_demo
  ```

- **Data Versioning & Lineage Tracking:**
  ```bash
  python3 -m src.versioning.data_versioning
  ```

- **Model Training & Metric Evaluation:**
  ```bash
  python3 -m src.model.train_evaluation
  ```

- **Real-Time Recommendation Inference Query:**
  ```bash
  python3 -m src.model.inference
  ```

---

## Deliverable Reports & Artifacts

All mandatory deliverables are saved in the `reports/` folder:
1. `reports/data_quality_report.pdf` & `reports/data_quality_report.md`
2. `reports/model_performance_report.pdf` & `reports/model_performance_report.md`
3. `reports/figures/`:
   - `interaction_distribution.png`
   - `item_popularity.png`
   - `sparsity_heatmap.png`

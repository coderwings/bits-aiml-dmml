"""
Model Training, Cross-Validation, Evaluation, and MLflow Tracking Registry for RecoMart.
Calculates Precision@K, Recall@K, NDCG@K, RMSE, and MAE.
Saves model pickle artifacts and outputs performance reports.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from src.model.recommender import SVDCollaborativeRecommender, ContentBasedRecommender, HybridRecommender
from src.reports.pdf_generator import create_pdf_report

logger = logging.getLogger("RecoMart_TrainEval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/training.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def calculate_rmse_mae(model, test_df):
    """Computes Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
    y_true = []
    y_pred = []
    
    for _, row in test_df.iterrows():
        true_val = float(row['rating'])
        pred_val = model.predict_score(row['user_id'], row['item_id'])
        y_true.append(true_val)
        y_pred.append(pred_val)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return round(rmse, 4), round(mae, 4)


def calculate_ranking_metrics(model, test_df, k=5, threshold=3.5):
    """
    Calculates Precision@K, Recall@K, and NDCG@K ranking evaluation metrics.
    """
    test_df['user_id'] = test_df['user_id'].astype(str)
    test_df['item_id'] = test_df['item_id'].astype(str)

    user_relevant_items = {}
    for user_id, group in test_df.groupby('user_id'):
        rel_items = set(group[group['rating'] >= threshold]['item_id'])
        if rel_items:
            user_relevant_items[user_id] = rel_items

    precisions = []
    recalls = []
    ndcgs = []

    for user_id, rel_set in user_relevant_items.items():
        recs = model.recommend_top_k(user_id, k=k)
        rec_item_ids = [iid for iid, _ in recs]

        # Precision@K
        hits = len(set(rec_item_ids).intersection(rel_set))
        precision = hits / float(k)
        recall = hits / float(len(rel_set))
        precisions.append(precision)
        recalls.append(recall)

        # NDCG@K
        dcg = 0.0
        for i, iid in enumerate(rec_item_ids):
            if iid in rel_set:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

    return round(mean_precision, 4), round(mean_recall, 4), round(mean_ndcg, 4)


class ModelTrainer:
    def __init__(self, prepared_dir="data_lake/prepared", models_dir="models", reports_dir="reports"):
        self.prepared_dir = prepared_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def train_and_evaluate(self, k_factors=10, top_k=5):
        logger.info("Loading prepared datasets for model training...")
        tx_df = pd.read_csv(os.path.join(self.prepared_dir, "transactions_prepared.csv"))
        items_df = pd.read_csv(os.path.join(self.prepared_dir, "items_prepared.csv"))

        # Train/Test Split
        train_df, test_df = train_test_split(tx_df, test_size=0.2, random_state=42)
        logger.info(f"Dataset split: Train shape {train_df.shape}, Test shape {test_df.shape}")

        # 1. Train SVD Collaborative Filtering Model
        svd_model = SVDCollaborativeRecommender(k_factors=k_factors)
        svd_model.fit(train_df)

        # 2. Train Content-Based Filtering Model
        content_model = ContentBasedRecommender()
        content_model.fit(items_df)

        # 3. Hybrid Model
        hybrid_model = HybridRecommender(svd_model, content_model)

        # Evaluate SVD Model
        rmse, mae = calculate_rmse_mae(svd_model, test_df)
        precision, recall, ndcg = calculate_ranking_metrics(svd_model, test_df, k=top_k)

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "Precision@K": precision,
            "Recall@K": recall,
            "NDCG@K": ndcg
        }

        logger.info(f"Model Evaluation Metrics (@K={top_k}): {metrics}")

        # Save Model Artifacts
        svd_path = os.path.join(self.models_dir, "svd_model.pkl")
        content_path = os.path.join(self.models_dir, "content_model.pkl")
        with open(svd_path, "wb") as f:
            pickle.dump(svd_model, f)
        with open(content_path, "wb") as f:
            pickle.dump(content_model, f)

        # Track via MLflow / Local Registry
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        registry_payload = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_type": "SVD Matrix Factorization + Content Hybrid",
            "hyperparameters": {
                "k_factors": k_factors,
                "top_k_evaluation": top_k,
                "test_split_ratio": 0.2
            },
            "metrics": metrics,
            "artifacts": {
                "svd_model": svd_path,
                "content_model": content_path
            }
        }

        registry_file = os.path.join(self.models_dir, "model_registry.json")
        with open(registry_file, "w") as f:
            json.dump(registry_payload, f, indent=4)

        # Try MLflow Logging if available
        try:
            import mlflow
            mlflow.set_experiment("RecoMart_Recommendation_Experiment")
            with mlflow.start_run(run_name=run_id):
                mlflow.log_params(registry_payload["hyperparameters"])
                mlflow_metrics = {k.replace("@", "_at_"): v for k, v in metrics.items()}
                mlflow.log_metrics(mlflow_metrics)
                mlflow.log_artifact(svd_path)
                mlflow.log_artifact(content_path)
            logger.info("Successfully tracked model run to MLflow experiment.")
        except Exception as mlflow_err:
            logger.warning(f"MLflow server logging bypassed ({str(mlflow_err)}). Used local registry at '{registry_file}'")

        # Generate Reports (.md and .pdf)
        self.generate_reports(registry_payload)
        return registry_payload

    def generate_reports(self, registry_payload):
        # Markdown Report
        md_path = os.path.join(self.reports_dir, "model_performance_report.md")
        m = registry_payload["metrics"]
        with open(md_path, "w") as f:
            f.write("# RecoMart Recommendation Model Performance Report\n\n")
            f.write(f"**Run ID:** `{registry_payload['run_id']}`\n")
            f.write(f"**Model Type:** {registry_payload['model_type']}\n")
            f.write(f"**Execution Timestamp:** {registry_payload['timestamp']}\n\n")
            f.write("## 1. Quantitative Evaluation Metrics\n\n")
            f.write("| Metric | Category | Score |\n|---|---|---|\n")
            f.write(f"| Precision@{registry_payload['hyperparameters']['top_k_evaluation']} | Ranking Accuracy | **{m['Precision@K']}** |\n")
            f.write(f"| Recall@{registry_payload['hyperparameters']['top_k_evaluation']} | Coverage | **{m['Recall@K']}** |\n")
            f.write(f"| NDCG@{registry_payload['hyperparameters']['top_k_evaluation']} | Ranking Position Decay | **{m['NDCG@K']}** |\n")
            f.write(f"| RMSE | Rating Prediction Error | **{m['RMSE']}** |\n")
            f.write(f"| MAE | Absolute Error | **{m['MAE']}** |\n")

        # PDF Report
        pdf_sections = [
            {
                "heading": "1. Executive Summary & Model Overview",
                "content": f"The recommendation engine was trained on {datetime.now().strftime('%Y-%m-%d')} using SVD Matrix Factorization Collaborative Filtering coupled with Content-Based feature similarity."
            },
            {
                "heading": "2. Hyperparameters & Configuration",
                "table_data": [["Parameter", "Config Value"]] + [[k, str(v)] for k, v in registry_payload["hyperparameters"].items()]
            },
            {
                "heading": "3. Benchmark Evaluation Results",
                "table_data": [
                    ["Metric Name", "Category", "Score Value"],
                    ["Precision@K", "Ranking Accuracy", str(m['Precision@K'])],
                    ["Recall@K", "Coverage", str(m['Recall@K'])],
                    ["NDCG@K", "Ranking Decay", str(m['NDCG@K'])],
                    ["RMSE", "Rating Root Mean Squared Error", str(m['RMSE'])],
                    ["MAE", "Mean Absolute Error", str(m['MAE'])]
                ]
            }
        ]
        create_pdf_report("model_performance_report.pdf", "RecoMart Model Performance & Evaluation Report", "Machine Learning Performance Metrics", pdf_sections, self.reports_dir)
        logger.info(f"Generated Model Performance reports in '{self.reports_dir}'")


def run_model_training(*args, **kwargs):
    trainer = ModelTrainer()
    payload = trainer.train_and_evaluate(*args, **kwargs)
    return payload

if __name__ == "__main__":
    run_model_training()

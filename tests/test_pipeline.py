"""
Unit and Integration Tests for RecoMart Recommendation System Data Management Pipeline.
Runs via pytest.
"""

import os
import pytest
import pandas as pd
from input_data.generate_synthetic_data import generate_synthetic_data
from src.ingestion.data_ingestion import run_ingestion
from src.validation.data_validation import DataValidator
from src.preparation.data_preparation import DataPreparer
from src.features.feature_engineering import run_feature_engineering
from src.features.feature_store import RecoMartFeatureStore
from src.model.train_evaluation import run_model_training
from src.model.inference import RecommendationInferenceService


@pytest.fixture(scope="module", autouse=True)
def setup_environment():
    """Ensure synthetic data is generated before tests run."""
    generate_synthetic_data(num_users=50, num_items=20, num_interactions=300)


def test_data_ingestion():
    files = run_ingestion()
    assert os.path.exists(files["transactions_file"])
    assert os.path.exists(files["users_file"])
    assert os.path.exists(files["metadata_file"])


def test_data_validation():
    validator = DataValidator()
    tx_report, tx_df = validator.validate_transactions()
    meta_report, meta_df = validator.validate_metadata()
    
    assert "schema_check" in tx_report["checks"]
    assert "purchase_value_check" in tx_report["checks"]
    assert tx_df is not None
    assert meta_df is not None


def test_data_preparation():
    preparer = DataPreparer()
    tx_clean = preparer.clean_transactions()
    items_clean = preparer.clean_items()
    users_clean = preparer.clean_users()

    assert not tx_clean.empty
    assert "implicit_rating" in tx_clean.columns
    assert "price_normalized" in items_clean.columns
    assert "tier_code" in users_clean.columns


def test_feature_engineering_and_store():
    run_feature_engineering()
    fs = RecoMartFeatureStore()
    
    # Online point lookup
    user_feat = fs.get_online_features("user", "1001")
    assert user_feat is not None
    assert "user_interaction_count" in user_feat

    # Batch historical join
    tx_sample = pd.DataFrame([{"user_id": "1001", "item_id": "501", "rating": 4.0}])
    joined = fs.get_historical_features(tx_sample)
    assert not joined.empty
    assert "price_normalized" in joined.columns


def test_model_training_and_inference():
    payload = run_model_training(k_factors=5, top_k=3)
    assert "metrics" in payload
    assert "RMSE" in payload["metrics"]
    assert payload["metrics"]["RMSE"] >= 0.0

    service = RecommendationInferenceService()
    recs = service.get_recommendations("1001", top_k=3)
    assert "top_k_recommendations" in recs
    assert len(recs["top_k_recommendations"]) <= 3

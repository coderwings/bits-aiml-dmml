# RecoMart Feature Store Architecture & Retrieval Interface

## Feature Views & Registry
The RecoMart Feature Store acts as a centralized metadata registry and relational feature warehouse storing versioned feature views:

1. **`user_features` View:** User entity features updated in offline batch runs.
2. **`item_features` View:** Product entity catalog features.
3. **`item_cooccurrence` View:** Item-item similarity scores for real-time cross-selling.

## Dual Serving Modes

### 1. Online Real-Time Inference Lookup
Low-latency single-entity lookup for model serving:
```python
from src.features.feature_store import RecoMartFeatureStore
fs = RecoMartFeatureStore()
user_vector = fs.get_online_features(entity_type="user", entity_id="1001")
```

### 2. Offline Batch Training Feature Join
Point-in-time join matching user and item feature tables to transaction logs with zero target leakage:
```python
historical_df = fs.get_historical_features(transactions_df)
```

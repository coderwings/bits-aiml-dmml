-- RecoMart Feature Store Database Schema Definition
-- Relational SQLite Warehouse tables for User, Item, and Interaction Features

CREATE TABLE IF NOT EXISTS user_features (
    user_id TEXT PRIMARY KEY,
    user_interaction_count INTEGER DEFAULT 0,
    user_avg_rating REAL DEFAULT 0.0,
    user_rating_std REAL DEFAULT 0.0,
    user_favorite_category TEXT,
    last_active_timestamp TEXT,
    tier_code INTEGER DEFAULT 1,
    feature_version TEXT DEFAULT 'v1.0'
);

CREATE TABLE IF NOT EXISTS item_features (
    item_id TEXT PRIMARY KEY,
    item_interaction_count INTEGER DEFAULT 0,
    item_avg_rating REAL DEFAULT 0.0,
    item_rating_std REAL DEFAULT 0.0,
    price_normalized REAL DEFAULT 0.0,
    sentiment_score REAL DEFAULT 0.0,
    popularity_normalized REAL DEFAULT 0.0,
    category_code INTEGER DEFAULT 0,
    feature_version TEXT DEFAULT 'v1.0'
);

CREATE TABLE IF NOT EXISTS item_cooccurrence (
    item_id_a TEXT,
    item_id_b TEXT,
    cooccurrence_count INTEGER,
    jaccard_similarity REAL,
    feature_version TEXT DEFAULT 'v1.0',
    PRIMARY KEY (item_id_a, item_id_b)
);

CREATE TABLE IF NOT EXISTS feature_store_metadata (
    feature_view_name TEXT PRIMARY KEY,
    entity_id TEXT,
    features_list TEXT,
    source_table TEXT,
    created_at TEXT,
    version TEXT
);

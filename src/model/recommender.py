"""
RecoMart Recommendation Algorithms Implementation.
Includes Collaborative Filtering (SVD Matrix Factorization), Content-Based Filtering, and Hybrid Ensembles.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("RecoMart_Recommender")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/model.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class SVDCollaborativeRecommender:
    """Matrix Factorization SVD Collaborative Filtering Model."""
    def __init__(self, k_factors=10):
        self.k_factors = k_factors
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.predicted_matrix = None
        self.global_mean = 0.0

    def fit(self, tx_df):
        logger.info("Training SVD Matrix Factorization Collaborative Filtering model...")
        unique_users = sorted(tx_df['user_id'].astype(str).unique())
        unique_items = sorted(tx_df['item_id'].astype(str).unique())
        
        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_map.items()}

        self.global_mean = tx_df['rating'].mean()
        
        # Build User-Item Interaction Matrix
        R = np.zeros((len(unique_users), len(unique_items)))
        for _, row in tx_df.iterrows():
            u_idx = self.user_map[str(row['user_id'])]
            i_idx = self.item_map[str(row['item_id'])]
            R[u_idx, i_idx] = float(row['implicit_rating'] if 'implicit_rating' in row else row['rating'])

        # Subtract user mean rating to normalize
        user_ratings_mean = np.array([row[row > 0].mean() if np.any(row > 0) else self.global_mean for row in R])
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)

        # Compute SVD Matrix Decomposition
        k = min(self.k_factors, min(R.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(R_demeaned), k=k)
        sigma = np.diag(sigma)
        
        self.predicted_matrix = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        logger.info(f"SVD training completed. Latent factors K={k}, Matrix shape: {self.predicted_matrix.shape}")
        return self

    def predict_score(self, user_id, item_id):
        u_str, i_str = str(user_id), str(item_id)
        if u_str not in self.user_map or i_str not in self.item_map:
            return self.global_mean
        u_idx = self.user_map[u_str]
        i_idx = self.item_map[i_str]
        return float(np.clip(self.predicted_matrix[u_idx, i_idx], 1.0, 5.0))

    def recommend_top_k(self, user_id, k=5, exclude_interacted=True, tx_df=None):
        u_str = str(user_id)
        if u_str not in self.user_map:
            # Fallback to global top items with global mean rating
            return [(iid, float(self.global_mean)) for iid in list(self.item_map.keys())[:k]]

        u_idx = self.user_map[u_str]
        user_scores = self.predicted_matrix[u_idx]

        interacted_items = set()
        if exclude_interacted and tx_df is not None:
            user_tx = tx_df[tx_df['user_id'].astype(str) == u_str]
            interacted_items = set(user_tx['item_id'].astype(str))

        item_indices = np.argsort(user_scores)[::-1]
        recs = []
        for idx in item_indices:
            iid = self.reverse_item_map[idx]
            if iid not in interacted_items:
                recs.append((iid, float(np.clip(user_scores[idx], 1.0, 5.0))))
            if len(recs) >= k:
                break
        return recs


class ContentBasedRecommender:
    """Item Feature Cosine Similarity Content-Based Filtering Model."""
    def __init__(self):
        self.items_df = None
        self.similarity_matrix = None
        self.item_map = {}
        self.reverse_item_map = {}

    def fit(self, items_df):
        logger.info("Training Content-Based Recommendation model on item features...")
        self.items_df = items_df.copy()
        self.items_df['item_id'] = self.items_df['item_id'].astype(str)
        
        feature_cols = ['price_normalized', 'sentiment_score', 'popularity_normalized', 'category_code']
        X = self.items_df[feature_cols].fillna(0.0).values
        
        self.similarity_matrix = cosine_similarity(X)
        self.item_map = {iid: idx for idx, iid in enumerate(self.items_df['item_id'])}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_map.items()}
        logger.info(f"Content-Based similarity matrix built for {len(self.items_df)} items.")
        return self

    def recommend_similar_items(self, item_id, k=5):
        i_str = str(item_id)
        if i_str not in self.item_map:
            return []
        
        i_idx = self.item_map[i_str]
        scores = self.similarity_matrix[i_idx]
        sorted_indices = np.argsort(scores)[::-1]
        
        recs = []
        for idx in sorted_indices:
            other_iid = self.reverse_item_map[idx]
            if other_iid != i_str:
                recs.append((other_iid, float(scores[idx])))
            if len(recs) >= k:
                break
        return recs


class HybridRecommender:
    """Ensemble Hybrid Combining Collaborative SVD and Content Similarity."""
    def __init__(self, svd_model, content_model, alpha=0.7):
        self.svd_model = svd_model
        self.content_model = content_model
        self.alpha = alpha  # Weight for SVD score

    def recommend_top_k(self, user_id, k=5, tx_df=None):
        svd_recs = dict(self.svd_model.recommend_top_k(user_id, k=k*2, tx_df=tx_df))
        if not svd_recs:
            return []

        # Get top interacted item for content similarity boost
        u_str = str(user_id)
        hybrid_scores = {}
        
        for iid, svd_score in svd_recs.items():
            # Normalized SVD rating (1-5 to 0-1)
            norm_svd = (svd_score - 1.0) / 4.0
            
            # Content similarity boost if user interacted with items
            content_boost = 0.5
            if tx_df is not None:
                user_tx = tx_df[tx_df['user_id'].astype(str) == u_str]
                if not user_tx.empty:
                    last_item = str(user_tx.iloc[-1]['item_id'])
                    similar = dict(self.content_model.recommend_similar_items(last_item, k=20))
                    content_boost = similar.get(iid, 0.0)

            final_score = self.alpha * norm_svd + (1.0 - self.alpha) * content_boost
            hybrid_scores[iid] = round(float(final_score), 4)

        sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return sorted_recs

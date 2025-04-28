"""
Utility functions for Retailrocket Hybrid Recommender System.

- Load saved models and encoders
- Generate hybrid recommendations
- Re-rank using LTR model
"""

import os
import numpy as np
import pickle
import faiss
import pandas as pd
from src.ltr_model import load_ltr_model, rerank_candidates

def load_models(model_dir='./models/'):
    """
    Load all saved models: encoders, factors, FAISS index, and LTR model.

    Args:
        model_dir (str): Path to saved models.

    Returns:
        dict: Dictionary containing loaded models and encoders.
    """
    # Load encoders
    user_encoder = pickle.load(open(os.path.join(model_dir, 'user_encoder.pkl'), 'rb'))
    item_encoder = pickle.load(open(os.path.join(model_dir, 'item_encoder.pkl'), 'rb'))

    # Load CF factors
    user_factors = np.load(os.path.join(model_dir, 'user_factors.npy'))
    item_factors_cf = np.load(os.path.join(model_dir, 'item_factors.npy'))

    # Load Content-Based FAISS index
    item_content_index = faiss.read_index(os.path.join(model_dir, 'item_content_index.faiss'))

    # Load LTR model
    ltr_model = load_ltr_model(model_dir)

    return {
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'user_factors': user_factors,
        'item_factors_cf': item_factors_cf,
        'item_content_index': item_content_index,
        'ltr_model': ltr_model
    }


def generate_recommendations(user_id, models, top_k=10, alpha=0.6):
    """
    Generate top-K product recommendations for a given user.

    Args:
        user_id (int or str): User ID (original, not encoded).
        models (dict): Loaded models and encoders.
        top_k (int): Number of items to recommend.
        alpha (float): Weight for CF vs Content in hybrid scoring.

    Returns:
        list: List of recommended item IDs (original, not encoded).
    """
    # Unpack models
    user_encoder = models['user_encoder']
    item_encoder = models['item_encoder']
    user_factors = models['user_factors']
    item_factors_cf = models['item_factors_cf']
    ltr_model = models['ltr_model']

    # Try to encode user
    if user_id not in user_encoder.classes_:
        return []

    user_idx = user_encoder.transform([user_id])[0]

    user_vector = user_factors[user_idx]

    # Hybrid scoring (CF + Content not FAISS here, using CF + dense hybrid for now)
    scores = np.dot(item_factors_cf, user_vector)

    # Get top 100 candidates first
    top_candidate_indices = np.argsort(scores)[::-1][:100]

    candidate_items = item_encoder.inverse_transform(top_candidate_indices)

    # Build candidate feature dataframe (simple)
    candidate_features = pd.DataFrame({
        'user_id_enc': user_idx,
        'item_id_enc': top_candidate_indices,
        'cf_score': scores[top_candidate_indices]
    })

    # Re-rank using LTR
    ranked_candidates = rerank_candidates(candidate_features, ltr_model)

    # Take top-K
    top_items = ranked_candidates['item_id_enc'].head(top_k).values
    top_item_ids = item_encoder.inverse_transform(top_items)

    return top_item_ids.tolist()
"""
Collaborative Filtering Model for Retailrocket Hybrid Recommender System.

- Train Collaborative Filtering model (Matrix Factorization)
- Save and load user/item embeddings
"""

import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
import os


def train_cf_model(interaction_matrix, n_components=50, save_dir='./models/'):
    """
    Train Collaborative Filtering model using Truncated SVD (Matrix Factorization Approximation).

    Args:
        interaction_matrix (csr_matrix): User-item interaction sparse matrix.
        n_components (int): Number of latent factors.
        save_dir (str): Directory to save model artifacts.

    Returns:
        user_factors (np.ndarray): User latent factor matrix.
        item_factors (np.ndarray): Item latent factor matrix.
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    print(f"Training Collaborative Filtering model with {n_components} latent factors...")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(interaction_matrix)
    item_factors = svd.components_.T  # transpose to get item latent factors

    # Save the factors
    np.save(os.path.join(save_dir, 'user_factors.npy'), user_factors)
    np.save(os.path.join(save_dir, 'item_factors.npy'), item_factors)

    print(f"User and item factors saved to {save_dir}")

    return user_factors, item_factors


def load_cf_model(save_dir='./models/'):
    """
    Load trained Collaborative Filtering user and item factors.

    Args:
        save_dir (str): Directory where factors are saved.

    Returns:
        user_factors (np.ndarray): User latent factor matrix.
        item_factors (np.ndarray): Item latent factor matrix.
    """
    user_factors = np.load(os.path.join(save_dir, 'user_factors.npy'))
    item_factors = np.load(os.path.join(save_dir, 'item_factors.npy'))

    return user_factors, item_factors


def predict_cf_score(user_idx, item_idx, user_factors, item_factors):
    """
    Predict score between a user and an item using dot product of latent factors.

    Args:
        user_idx (int): Encoded user index.
        item_idx (int): Encoded item index.
        user_factors (np.ndarray): User latent factors.
        item_factors (np.ndarray): Item latent factors.

    Returns:
        float: Predicted interaction score.
    """
    return np.dot(user_factors[user_idx], item_factors[item_idx])
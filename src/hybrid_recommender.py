"""
Hybrid Recommender System for Retailrocket Hybrid Project.

- Combine Collaborative Filtering and Content-Based Filtering scores
- Weighted fusion approach for final recommendation ranking
"""

import numpy as np

def compute_hybrid_score(user_vector, item_vector_cf, item_vector_content, alpha=0.5):
    """
    Compute a hybrid recommendation score by combining CF and Content-Based similarities.

    Args:
        user_vector (np.ndarray): Latent factors for the user (from CF model).
        item_vector_cf (np.ndarray): Latent factors for the item (from CF model).
        item_vector_content (np.ndarray): Content feature vector for the item.
        alpha (float): Weight for CF (0.0 - 1.0). (1-alpha) is weight for content-based.

    Returns:
        float: Final hybrid recommendation score.
    """
    cf_score = np.dot(user_vector, item_vector_cf)
    content_score = np.dot(user_vector, item_vector_content)

    hybrid_score = alpha * cf_score + (1 - alpha) * content_score
    return hybrid_score


def rank_items_for_user(user_idx, user_factors, item_factors_cf, item_factors_content, top_k=10, alpha=0.5):
    """
    Generate top-K ranked item recommendations for a given user.

    Args:
        user_idx (int): Encoded user index.
        user_factors (np.ndarray): User latent factors (CF model).
        item_factors_cf (np.ndarray): Item latent factors (CF model).
        item_factors_content (np.ndarray): Item feature matrix (content-based).
        top_k (int): Number of items to recommend.
        alpha (float): Weight for CF vs Content-Based.

    Returns:
        list: List of item indices ranked by hybrid score.
    """
    user_vector = user_factors[user_idx]

    # Compute hybrid scores for all items
    scores = np.zeros(item_factors_cf.shape[0])

    for item_idx in range(item_factors_cf.shape[0]):
        scores[item_idx] = compute_hybrid_score(
            user_vector,
            item_factors_cf[item_idx],
            item_factors_content[item_idx],
            alpha=alpha
        )

    # Get top-K items
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]
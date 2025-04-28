"""
Learning-to-Rank (LTR) Model for Retailrocket Hybrid Recommender System.

- Train a LambdaRank model using LightGBM
- Save and load LTR model
- Apply LTR model to re-rank top candidate recommendations
"""

import os
import lightgbm as lgb
import pandas as pd
import numpy as np

def train_ltr(features_df, save_dir='./models/', num_boost_round=100):
    """
    Train a LightGBM Learning-to-Rank (LTR) model.

    Args:
        features_df (pd.DataFrame): DataFrame with columns: ['user_id_enc', 'item_id_enc', feature_cols..., 'label']
        save_dir (str): Directory to save the trained LTR model.
        num_boost_round (int): Number of boosting rounds.

    Returns:
        booster: Trained LightGBM model.
    """
    os.makedirs(save_dir, exist_ok=True)

    feature_cols = [col for col in features_df.columns if col not in ['user_id_enc', 'item_id_enc', 'label']]

    X = features_df[feature_cols]
    y = features_df['label']
    group = features_df.groupby('user_id_enc').size().values  # Important for ranking tasks

    lgb_train = lgb.Dataset(X, label=y, group=group)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 30,
        'verbose': -1
    }

    print("Training LTR model...")
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_boost_round,
    )

    # Save model
    model_path = os.path.join(save_dir, 'ltr_model.txt')
    booster.save_model(model_path)
    print(f"LTR model saved to {model_path}")

    return booster


def load_ltr_model(save_dir='./models/'):
    """
    Load a trained LightGBM Learning-to-Rank model.

    Args:
        save_dir (str): Directory where the model is saved.

    Returns:
        booster: Loaded LightGBM model.
    """
    model_path = os.path.join(save_dir, 'ltr_model.txt')
    assert os.path.exists(model_path), f"LTR model file not found at {model_path}"

    booster = lgb.Booster(model_file=model_path)
    return booster


def rerank_candidates(features_df, ltr_model):
    """
    Re-rank candidate items for a user based on LTR model scores.

    Args:
        features_df (pd.DataFrame): DataFrame of candidate features (same columns as during training).
        ltr_model (booster): Trained LightGBM LTR model.

    Returns:
        pd.DataFrame: DataFrame with additional column 'ltr_score', sorted by score descending.
    """
    feature_cols = [col for col in features_df.columns if col not in ['user_id_enc', 'item_id_enc', 'label']]

    X = features_df[feature_cols]
    features_df['ltr_score'] = ltr_model.predict(X)

    ranked_df = features_df.sort_values('ltr_score', ascending=False).reset_index(drop=True)
    return ranked_df
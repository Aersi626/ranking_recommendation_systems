"""
Data preprocessing functions for Retailrocket Hybrid Recommender System.

- Load Retailrocket datasets
- Filter active users and popular items
- Prepare data for Collaborative Filtering and Content-Based Filtering
"""

import pandas as pd
import numpy as np
import os

def preprocess_data(events_path='./data/filtered_events.csv',
                    item_properties_path='./data/item_properties_part1.csv',
                    min_user_interactions=5,
                    min_item_interactions=10):
    """
    Preprocess the Retailrocket dataset:
    - Load event data
    - Filter for active users and popular items
    - Prepare for modeling

    Args:
        events_path (str): Path to filtered events CSV.
        item_properties_path (str): Path to item properties CSV.
        min_user_interactions (int): Minimum events per user to keep.
        min_item_interactions (int): Minimum events per item to keep.

    Returns:
        events (pd.DataFrame): Cleaned events data.
        item_properties (pd.DataFrame): Item metadata.
    """

    # Ensure data paths exist
    assert os.path.exists(events_path), f"Events file not found at {events_path}"
    assert os.path.exists(item_properties_path), f"Item properties file not found at {item_properties_path}"

    # Load datasets
    events = pd.read_csv(events_path)
    item_properties = pd.read_csv(item_properties_path)

    print(f"Loaded events: {events.shape}, item_properties: {item_properties.shape}")

    # Filter active users
    user_counts = events['visitorid'].value_counts()
    active_users = user_counts[user_counts >= min_user_interactions].index
    events = events[events['visitorid'].isin(active_users)]

    # Filter popular items
    item_counts = events['itemid'].value_counts()
    popular_items = item_counts[item_counts >= min_item_interactions].index
    events = events[events['itemid'].isin(popular_items)]

    print(f"After filtering: {events.shape}")

    # Reset index for clean processing
    events = events.reset_index(drop=True)

    return events, item_properties


def generate_user_item_matrix(events):
    """
    Create a user-item interaction matrix.

    Args:
        events (pd.DataFrame): Event data after preprocessing.

    Returns:
        interaction_matrix (scipy.sparse.csr_matrix): User-item interaction matrix.
        user_encoder (LabelEncoder): User ID encoder.
        item_encoder (LabelEncoder): Item ID encoder.
    """
    from sklearn.preprocessing import LabelEncoder
    from scipy.sparse import csr_matrix

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    events['user_id_enc'] = user_encoder.fit_transform(events['visitorid'])
    events['item_id_enc'] = item_encoder.fit_transform(events['itemid'])

    n_users = events['user_id_enc'].nunique()
    n_items = events['item_id_enc'].nunique()

    interaction_matrix = csr_matrix(
        (1 * np.ones(events.shape[0]), (events['user_id_enc'], events['item_id_enc'])),
        shape=(n_users, n_items)
    )

    return interaction_matrix, user_encoder, item_encoder
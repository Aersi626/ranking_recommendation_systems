"""
Content-Based Filtering Model for Retailrocket Hybrid Recommender System.

- Build item feature embeddings
- Create FAISS index for fast item similarity search
- Save and load FAISS index
"""

import pandas as pd
import numpy as np
import faiss
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD



def train_cb_model(item_properties_path='./data/item_properties_part1.csv',
                   save_dir='./models/',
                   top_n_features=50):
    """
    Train a simple Content-Based Filtering model using item metadata.

    Args:
        item_properties_path (str): Path to item properties CSV.
        save_dir (str): Directory to save FAISS index.
        top_n_features (int): Number of top property types to keep.

    Returns:
        faiss.IndexFlatL2: Trained FAISS index for item similarity search.
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load item properties
    item_properties = pd.read_csv(item_properties_path)
    item_properties = item_properties.dropna()

    # Take latest properties for each item
    item_properties_latest = item_properties.sort_values('timestamp').drop_duplicates('itemid', keep='last')

    # Pivot properties to wide format
    item_metadata = item_properties_latest.pivot_table(
        index='itemid', columns='property', values='value', aggfunc='first'
    )
    item_metadata = item_metadata.fillna('unknown')

    # Use only the top-N most frequent properties (optional)
    if top_n_features:
        property_counts = item_properties['property'].value_counts().head(top_n_features).index
        item_metadata = item_metadata[property_counts]

    # # One-hot encode item features
    item_metadata = item_metadata.applymap(str)  # Ensure all features are strings
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)
    item_features = encoder.fit_transform(item_metadata)

    # item features could be huge
    n_components = 100  
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    item_features_reduced = svd.fit_transform(item_features)

    # now build FAISS on the reduced, dense array
    item_features_reduced = item_features_reduced.astype('float32')
    index = faiss.IndexFlatL2(n_components)
    index.add(item_features_reduced)

    # Save FAISS index
    faiss.write_index(index, os.path.join(save_dir, 'item_content_index.faiss'))

    print(f"Content-Based Filtering model trained and FAISS index saved to {save_dir}.")

    return index


def load_cb_model(save_dir='./models/'):
    """
    Load saved FAISS content-based index.

    Args:
        save_dir (str): Directory containing saved FAISS index.

    Returns:
        faiss.IndexFlatL2: Loaded FAISS index.
    """
    index_path = os.path.join(save_dir, 'item_content_index.faiss')
    assert os.path.exists(index_path), f"FAISS index not found at {index_path}"

    index = faiss.read_index(index_path)
    return index


def get_similar_items(query_item_feature, index, top_k=10):
    """
    Retrieve top-K most similar items for a given item feature vector.

    Args:
        query_item_feature (np.ndarray): Feature vector of query item (1D).
        index (faiss.IndexFlatL2): Trained FAISS index.
        top_k (int): Number of neighbors to retrieve.

    Returns:
        indices (np.ndarray): Indices of top-K similar items.
        distances (np.ndarray): Distances to top-K similar items.
    """
    query = np.expand_dims(query_item_feature, axis=0)  # Make it 2D
    distances, indices = index.search(query, top_k)
    return indices[0], distances[0]
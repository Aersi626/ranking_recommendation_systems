"""
Training pipeline for Retailrocket Hybrid Recommender System.

- Preprocess data
- Train Collaborative Filtering model
- Train Content-Based Filtering model
- Train Learning-to-Rank (LTR) model
- Save models to /models/
"""

from src.data_preprocessing import preprocess_data, generate_user_item_matrix
from src.collaborative_filtering import train_cf_model
from src.content_based_filtering import train_cb_model
from src.ltr_model import train_ltr

import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    print("Starting training pipeline..")

    # 1. Preprocess Data
    events, item_properties = preprocess_data()
    interaction_matrix, user_encoder, item_encoder = generate_user_item_matrix(events)

    # Save Encoders
    os.makedirs('./models', exist_ok=True)
    import pickle
    pickle.dump(user_encoder, open('./models/user_encoder.pkl', 'wb'))
    pickle.dump(item_encoder, open('./models/item_encoder.pkl', 'wb'))

    print("Data preprocessed and encoders saved.")

    # 2. Train Collaborative Filtering Model
    user_factors, item_factors_cf = train_cf_model(interaction_matrix)

    # 3. Train Content-Based Model
    content_index = train_cb_model()

    # 4. Prepare Features for LTR Training
    # For now, generate simple features (later you can make this fancier)
    user_ids = events['visitorid'].unique()
    feature_rows = []

    for user in user_ids:
        try:
            encoded_user = user_encoder.transform([user])[0]
        except:
            continue
        # Predict scores for all items (only CF score for simplicity now)
        scores = np.dot(item_factors_cf, user_factors[encoded_user])

        top_items_idx = np.argsort(scores)[::-1][:50]  # Top 50 candidates
        for item_idx in top_items_idx:
            feature_rows.append({
                'user_id_enc': encoded_user,
                'item_id_enc': item_idx,
                'cf_score': scores[item_idx],
                'label': 1  # Pretend top items are relevant (can improve later)
            })

    features_df = pd.DataFrame(feature_rows)

    # 5. Train LTR Model
    ltr_model = train_ltr(features_df)

    print("Training pipeline completed successfully. All models saved!")
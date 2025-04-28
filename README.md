🛒 RetailRocket Hybrid Recommender System

This project implements a production-grade recommendation engine using the RetailRocket dataset.
It combines Collaborative Filtering, Content-Based Filtering, and a Learning-to-Rank (LTR) model to deliver personalized product recommendations through a FastAPI service.

🔥 Key Technologies Used
	•	FastAPI — Lightweight API serving
	•	LightGBM — LambdaRank Learning-to-Rank model
	•	FAISS — Approximate nearest neighbor search for content-based retrieval
	•	Scikit-learn — SVD dimensionality reduction
	•	Docker — Containerized deployment
	•	Python 3.10 — Development environment
	•	RetailRocket Dataset — Real-world e-commerce data


🎯 Future Improvements
	•	Add A/B testing framework for model comparison.
	•	Implement online feature store for real-time inference.
	•	Integrate more metadata into content-based modeling.
	•	Explore Neural Collaborative Filtering or Transformers for improved recommendations.
"""
FastAPI app to serve Retailrocket Hybrid Recommendations.

POST /recommend
Input: user_id
Output: Top-K product recommendations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.utils import load_models, generate_recommendations
from fastapi.responses import JSONResponse
import pickle

# Initialize FastAPI app
app = FastAPI(
    title="Retailrocket Hybrid Recommender API",
    description="Serve personalized recommendations using Hybrid + LTR models",
    version="1.0"
)

# Load models once at startup
models = load_models()

# Request schema
class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10

# Response schema
class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list

@app.get("/")
def root():
    return JSONResponse(content={"message": "Recommendation API is running! Use /recommend endpoint."})

@app.get("/recommend")
def recommend_instructions():
    return {"message": "Please send a POST request with JSON payload like {'user_id': 123, 'top_k': 10}"}

@app.get("/sample_user_ids")
def get_sample_user_ids(n: int = 10):
    with open("models/user_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    user_ids = encoder.classes_[:n]
    return {"user_ids": user_ids.tolist()}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Recommend top-K products for a given user ID.
    """
    user_id = request.user_id
    top_k = request.top_k

    recommendations = generate_recommendations(user_id, models, top_k)

    if not recommendations:
        raise HTTPException(status_code=404, detail="User ID not found or no recommendations available.")

    return RecommendResponse(
        user_id=user_id,
        recommendations=recommendations
    )



# Run server locally (for testing without docker-compose)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
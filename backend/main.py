from fastapi import FastAPI
import pandas as pd
import os
from vectorizer_loader import load_tfidf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load everything
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'tfidf_model.json')  # Now using JSON
DATA_PATH = os.path.join(DATA_DIR, 'recipes_final.pkl')

try:
    print("⏳ Loading data...")
    final_df = pd.read_pickle(DATA_PATH)
    
    print("⏳ Loading TF-IDF model...")
    tfidf = load_tfidf(MODEL_PATH)  # Using our new loader
    tfidf_matrix = tfidf.transform(final_df['NER_cleaned'])
    print("✅ Successfully loaded model")
    
except Exception as e:
    print(f"🔥 Error: {e}")
    raise

# 6. Recommendation function (unchanged from your notebook)
def find_similar_recipes(user_ingredients, top_n=3, penalty_weight=0.3):
    user_vec = tfidf.transform([user_ingredients])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix)
    
    user_ingredients_set = set(user_ingredients)
    penalty_scores = [
        len(user_ingredients_set - set(ingredients)) * penalty_weight
        for ingredients in final_df['NER_cleaned']
    ]
    
    adjusted_scores = cosine_sim[0] - np.array(penalty_scores)
    top_indices = np.argsort(adjusted_scores)[-top_n:][::-1]
    return final_df.iloc[top_indices]


from fastapi import FastAPI
from pydantic import BaseModel  # <-- Add this import

# Add this model definition
class IngredientsRequest(BaseModel):
    ingredients: list[str]

# 7. API Endpoints
@app.post("/suggest")
async def suggest(request: IngredientsRequest):  # <-- Use the model here
    try:
        results = find_similar_recipes(request.ingredients)  # <-- Access via request.ingredients
        return {
            "status": "success",
            "suggestions": results[['title', 'NER_cleaned']].to_dict(orient="records")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "recipe_count": len(final_df)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
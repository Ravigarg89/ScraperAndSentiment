from fastapi import FastAPI, Query
from pydantic import BaseModel
from tone_predictor import predict_tones
from typing import List
import subprocess
import json

app = FastAPI()

@app.get("/scrape")
def scrape_data(query: str = Query(...), city: str = Query(...)):
    """Scrapes info for a given query and city using external scraper_worker.py."""
    try:
        # Run the scraper_worker.py with query and city as arguments
        result = subprocess.run(
            ["python", "scraper.py", query, city],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return {"error": f"Scraper failed: {result.stderr.strip()}"}

        output = result.stdout.strip()

        # Try to parse output as JSON
        try:
            info = json.loads(output)
            return {"businesses": info}
        except json.JSONDecodeError:
            return {"error": "Invalid scraper output", "raw_output": output}

    except subprocess.TimeoutExpired:
        return {"error": "Scraper timed out"}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# POST endpoint for tone prediction
class ReviewInput(BaseModel):
    reviews: List[str]

@app.post("/predict")
def predict_tone(input: ReviewInput):
    predictions = predict_tones(input.reviews)
    return {"predictions": predictions}

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the scraper API. Available endpoints: /scrape (GET), /predict (POST). Visit /docs for interactive API."
    }

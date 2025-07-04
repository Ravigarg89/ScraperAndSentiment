import re
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

client = OpenAI(
    api_key="",  
    base_url="https://api.groq.com/openai/v1"
)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_API_BASE")
)


MODEL = "llama3-70b-8192"

def build_prompt(review, sentiment, tones):
    sorted_sentiments = sorted(sentiment.items(), key=lambda x: -x[1])
    sorted_tones = sorted(tones.items(), key=lambda x: -x[1])
    dominant_sentiment = sorted_sentiments[0][0]
    top_tones = [tone for tone, prob in sorted_tones[:2]]

    prompt = f"""
You are a professional service admin responding to customer feedback. 
Your goal is to generate 2–3 empathetic, polite, and helpful replies based on the review and emotional tone.

Review:
"{review}"

Dominant Sentiment: {dominant_sentiment}
Primary Tones: {", ".join(top_tones)}

Generate natural-sounding admin responses that:
- Acknowledge the issue
- Apologize if needed
- Reassure the customer or mention corrective action
- Do NOT copy the review directly
- Stay general enough for any service (bus, hotel, app, etc.)

Respond exactly in this format:

Response 1: ...
Response 2: ...
Response 3: ...
Do not include any introductory text or explanation.
"""
    return prompt.strip()

def generate_admin_response(review: str, sentiment_scores: dict, tones: dict) -> dict:
    prompt = build_prompt(review, sentiment_scores, tones)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a customer service admin replying professionally to user feedback."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=500
    )

    full_text = response.choices[0].message.content

  # Extract each numbered response directly using regex
    matches = re.findall(r"Response\s*\d+:\s*(.*?)(?=Response\s*\d+:|$)", full_text, flags=re.DOTALL)

    # Clean and build response dict
    response_dict = {f"Response {i+1}": match.strip().strip('"') for i, match in enumerate(matches)}

    return response_dict

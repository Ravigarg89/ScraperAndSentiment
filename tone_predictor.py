# tone_predictor.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from typing import List
import torch

# Load RoBERTa sentiment model (probability scores)
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Load emotion classification pipeline
emotion_classifier = pipeline("text-classification",
                              model="j-hartmann/emotion-english-distilroberta-base",
                              top_k = 3, framework="pt")

def polarity_scores_roberta(text: str) -> dict:
    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded)
    scores = softmax(output[0][0].numpy())
    return {
        'negative': float(scores[0]),
        'neutral': float(scores[1]),
        'positive': float(scores[2])
    }

def get_tones_from_textblock(textblock: str) -> dict:
    try:
        result = emotion_classifier(textblock[:512])
        if isinstance(result, list) and isinstance(result[0], list):
            return {
                r["label"]: round(float(r["score"]), 4)
                for r in result[0]
            }
        return {}
    except Exception as e:
        print(f"Tone extraction error: {e}")
        return {}


def predict_tones(reviews: List[str]) -> List[dict]:
    results = []
    for review in reviews:
        sentiment = polarity_scores_roberta(review)
        tone_labels = get_tones_from_textblock(review)
        results.append({
            "sentiment_scores": sentiment,
            "tones": tone_labels
        })
    return results


if __name__ == "__main__":
    print("tones")
    print(predict_tones(['Pathetic experience. AC & blower not working. All passengers are feeling very uncomfortable. Is bus me kabhi travel nahi karna boht gandi service hai.']))

import joblib
from src.preprocessing import clean_text

# Load saved model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("✅ Model & vectorizer loaded successfully!\n")

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction

# -----------------------
# Add your custom reviews
# -----------------------
new_reviews = [
    "This movie was absolutely amazing!",
    "Good content but the acting was very worst",
    "The story was okay, but acting was terrible.",
    "I loved the characters and the soundtrack."
]

for review in new_reviews:
    result = predict_sentiment(review)
    print(f"Review: {review}")
    print(f"Prediction: {result.upper()}\n")

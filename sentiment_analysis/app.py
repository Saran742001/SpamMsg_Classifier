from flask import Flask, request, jsonify
import joblib
from src.preprocessing import clean_text

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/")
def home():
    return {"message": "Sentiment Analysis API is running 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Invalid input"}), 400

    text = data["text"]

    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "Text must be non-empty"}), 400

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    confidence = round(max(probabilities), 3)

    return jsonify({
        "input_text": text,
        "sentiment": prediction,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)

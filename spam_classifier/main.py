import os
import joblib
from src.preprocessing import clean_text

# -------------------------------
# Load model & vectorizer
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "spam_nb_model.pkl")
)
vectorizer = joblib.load(
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
)

# -------------------------------
# Prediction function
# -------------------------------
def predict_message(message: str) -> str:
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "SPAM 🚨" if prediction == 1 else "HAM ✅"

# -------------------------------
# CLI Interface
# -------------------------------
def main():
    print("📩 Spam Classifier")
    print("Type 'exit' to quit\n")

    while True:
        message = input("Enter message: ")

        if message.lower() == "exit":
            print("👋 Goodbye!")
            break

        result = predict_message(message)
        print("Prediction:", result, "\n")


if __name__ == "__main__":
    main()

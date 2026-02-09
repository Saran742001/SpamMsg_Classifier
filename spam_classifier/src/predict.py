import os
import joblib
from src.preprocessing import clean_text   # ✅ FIXED IMPORT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "spam_nb_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))

def predict_message(message: str):
    cleaned_msg = clean_text(message)
    vector = vectorizer.transform([cleaned_msg])
    prediction = model.predict(vector)[0]
    return "SPAM 🚨" if prediction == 1 else "HAM ✅"

if __name__ == "__main__":
    print("📩 Spam Classifier (type 'exit' to quit)\n")

    while True:
        msg = input("Enter message: ")
        if msg.lower() == "exit":
            print("👋 Bye!")
            break

        print("Prediction:", predict_message(msg), "\n")

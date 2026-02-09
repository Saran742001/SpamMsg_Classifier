from src.load_data import load_dataset
from src.preprocessing import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib


def main():
    # ---------------------------
    # Load dataset
    # ---------------------------
    df = load_dataset("data/imdb_reviews.csv")

    # Preprocess text
    df["clean_review"] = df["review"].apply(clean_text)

    X = df["clean_review"]
    y = df["sentiment"]

    # ---------------------------
    # TF-IDF Vectorization
    # ---------------------------
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_tfidf = vectorizer.fit_transform(X)

    # ---------------------------
    # Train-test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------
    # Train model
    # ---------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # ---------------------------
    # Evaluate
    # ---------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("🎯 Model Accuracy:", accuracy)

    # ---------------------------
    # Save model & vectorizer
    # ---------------------------
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    print("✅ Model saved as sentiment_model.pkl")
    print("✅ Vectorizer saved as tfidf_vectorizer.pkl")

    # ===========================
    # 🔥 INSTANT USER PREDICTION
    # ===========================
    print("\n🧠 Sentiment Analyzer Ready!")
    print("Type a review to predict sentiment")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("📝 Enter review: ")

        if user_input.lower() == "exit":
            print("👋 Exiting Sentiment Analyzer")
            break

        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        print(f"🎯 Sentiment: {prediction.upper()}\n")


if __name__ == "__main__":
    main()

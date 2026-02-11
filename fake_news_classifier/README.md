# Fake News Classifier 🕵️‍♂️📰

This project is a Natural Language Processing (NLP) based application that classifies news articles as either **REAL** or **FAKE**. It leverages machine learning techniques to analyze text content and predict its authenticity.

## 🚀 Key Features

*   **Text Preprocessing Pipeline**: Efficient cleaning using NLTK (lowercasing, removing special characters, stopword removal, and Porter Stemming).
*   **TF-IDF Vectorization**: Converts text into numerical features using Term Frequency-Inverse Document Frequency.
*   **Machine Learning Model**: Uses **Logistic Regression** for binary classification (Real vs. Fake).
*   **Interactive CLI**: Train the model and test predictions directly from the command line.
*   **Flask API**: A RESTful API to serve the model for external applications.

---

## 📂 Project Structure

```bash
📦 fake_news_classifier
├── 📂 data
│   └── fake_news.csv          # Dataset file
├── 📂 src
│   ├── load_data.py           # Helper to load dataset
│   └── preprocessing.py       # Text cleaning and preprocessing logic
├── app.py                     # Flask API application
├── main.py                    # Training script and CLI interface
├── requirements.txt           # Python dependencies
├── fake_news_model.pkl        # Trained Logistic Regression model (generated after training)
└── tfidf_vectorizer.pkl       # Trained TF-IDF Vectorizer (generated after training)
```

---

## 🛠️ Installation

1.  **Clone the repository** (if applicable):
    ```bash
    git clone <repository-url>
    cd fake_news_classifier
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ Usage

### 1. Train the Model & Run CLI Prediction

The `main.py` script handles training the model, saving the artifacts (`2 .pkl files`), and launching an interactive prediction loop.

```bash
python main.py
```

**What happens:**
1.  The script loads `data/fake_news.csv`.
2.  Preprocessing is applied (Stemming, Stopword removal).
3.  TF-IDF Vectorization transforms the text.
4.  A Logistic Regression model is trained and evaluated.
5.  The model and vectorizer are saved locally.
6.  You enter an interactive loop to type news text and get real-time predictions.

### 2. Run the Flask API

To serve the model via a REST API:

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000/`.

#### API Endpoints:

*   **Health Check**: `GET /`
    *   Returns a welcome message verifying the API is running.

*   **Predict**: `POST /predict`
    *   **Body**: JSON object with a `text` field.
    *   **Data Example**:
        ```json
        {
          "text": "Breaking: Aliens have landed in New York City and are demanding pizza."
        }
        ```
    *   **Response Example**:
        ```json
        {
          "input_text": "Breaking: Aliens have landed...",
          "prediction": "FAKE"
        }
        ```

#### Example CURL Request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The government announced a new tax policy today aimed at reducing inflation."}'
```

---

## 🧠 Model Details

*   **Preprocessing**:
    *   Lowercasing
    *   Regex removal of non-alphabetic characters
    *   **NLTK Stopwords** removal
    *   **Porter Stemming**
*   **Vectorization**:
    *   `TfidfVectorizer` (Scikit-learn)
    *   `ngram_range=(1, 2)` (Unigrams and Bigrams)
    *   `max_features=5000`
*   **Classifier**:
    *   `LogisticRegression` (Scikit-learn)

---

## 📦 Dependencies

*   `flask`
*   `scikit-learn`
*   `pandas`
*   `numpy`
*   `nltk`
*   `joblib`

See `requirements.txt` for specific versions.

---

## ✍️ Author

*   **Saran742001** - *Initial work* - [GitHub Profile](https://github.com/Saran742001)

## 📅 Last Updated

*   **11th February 2026**

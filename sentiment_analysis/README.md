🎬 Sentiment Analysis using NLP & Machine Learning

A complete end-to-end NLP project that classifies movie reviews as Positive or Negative using TF-IDF + Logistic Regression, and exposes predictions through a Flask REST API with confidence scores.



📌 Project Highlights

Text preprocessing using NLP techniques

Feature extraction using TF-IDF

Sentiment classification using Logistic Regression

Model persistence using joblib

REST API built with Flask

Tested using Postman

Confidence score included in predictions





📂 Project Structure

sentiment_analysis/
│
├── data/
│   └── imdb_reviews.csv          # Dataset
│
├── src/
│   ├── load_data.py              # Dataset loader
│   ├── preprocessing.py          # Text cleaning utilities
│
├── app.py                        # Flask API
├── main.py                       # Model training script
├── sentiment_model.pkl           # Saved ML model
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation







🎯 Project Objective

The goal of this project is to:

Learn Natural Language Processing (NLP)

Build a real-world sentiment classification system

Deploy an ML model as a REST API

Enable real-time predictions via HTTP requests

🧠 Technologies Used
Technology	                    Purpose
Python	                    Core programming
Pandas	                    Data handling
NLTK	                    Text preprocessing
Scikit-learn                ML algorithms
TF-IDF	                    Feature extraction
Logistic Regression	        Classification
Flask	                    REST API
Joblib	                    Model saving/loading
Postman	                    API testing








📊 Dataset Information

Dataset: IMDb Movie Reviews

Columns:

    review → Movie review text

    sentiment → positive / negative

Size: ~50,000 reviews

Language: English








⚙️ Setup Instructions

1️⃣ Clone / Navigate to Project

cd sentiment_analysis



2️⃣ Create Virtual Environment

python3 -m venv .venv

source .venv/bin/activate



3️⃣ Install Dependencies

pip install -r requirements.txt



If requirements.txt not created yet:

pip install pandas scikit-learn nltk flask joblib



4️⃣ Download NLTK Stopwords

python -c "import nltk; nltk.download('stopwords')"

🏗️ Model Training

Run the training script:

python main.py



What Happens:

Dataset is loaded

Text is cleaned

TF-IDF features are generated

Logistic Regression model is trained

Accuracy is printed

Model & vectorizer are saved

Example Output:
🎯 Model Accuracy: 0.89
✅ Model saved as sentiment_model.pkl
✅ Vectorizer saved as tfidf_vectorizer.pkl




🔍 Text Preprocessing Steps

Performed in preprocessing.py:



Convert text to lowercase

Remove punctuation & special characters

Remove stopwords

Normalize spacing



🌐 Flask API

Start the API Server

python app.py




Server runs at:

http://127.0.0.1:5000

🔌 API Endpoints

✅ Health Check

GET /

http://127.0.0.1:5000/


Response:

{
  "message": "Sentiment Analysis API is running"
}

✅ Predict Sentiment

POST /predict

http://127.0.0.1:5000/predict

Request Body (JSON)

{
  "text": "This movie was absolutely amazing"
}


Response
{
  "input_text": "This movie was absolutely amazing",
  "sentiment": "positive",
  "confidence": 0.94
}




🧪 Testing with Postman
Steps:

Open Postman

Set method → POST

URL → http://127.0.0.1:5000/predict

Headers → Content-Type: application/json

Body → raw → JSON


Example:

{
  "text": "Worst movie I have ever seen"
}







📈 Model Details

Algorithm: Logistic Regression

Vectorizer: TF-IDF

Max Features: 5000

N-grams: Unigrams + Bigrams

Train/Test Split: 80 / 20


🧾 Output Explanation
Field	Meaning
sentiment	Final prediction
confidence	Model certainty
input_text	Original input





🚀 Learning Outcomes

✔ NLP preprocessing
✔ TF-IDF understanding
✔ Text classification
✔ Model evaluation
✔ REST API creation
✔ API testing using Postman





🛠️ Common Errors & Fixes
❌ NLTK stopwords error
python -c "import nltk; nltk.download('stopwords')"

❌ Model file not found

➡ Run python main.py before app.py

❌ Import errors

➡ Run commands from project root





🔮 Future Improvements (Optional)

Web UI using HTML/CSS

Deploy API to cloud

Add neutral sentiment

Switch to deep learning models

Dockerize the application






👤 Author

Saran
Built as part of NLP & Machine Learning learning journey.




📌 Project Status

✅ Completed up to API testing
🟡 Ready for deployment & frontend integration
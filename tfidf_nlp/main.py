import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = ENGLISH_STOP_WORDS


def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation, numbers, emojis
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize using regex (NO nltk punkt needed)
    tokens = text.split()

    # Remove stopwords & apply stemming
    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)


documents = [
    "I loved the movie! It was amazing and inspiring.",
    "The movie was boring and too long.",
    "Amazing acting, but the story was bad.",
    "I would not recommend this boring movie."
]

cleaned_documents = [clean_text(doc) for doc in documents]

print("CLEANED TEXT:")
for doc in cleaned_documents:
    print(doc)

vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(cleaned_documents)

print("\nTF-IDF FEATURES:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF VECTORS:")
print(tfidf_vectors.toarray())

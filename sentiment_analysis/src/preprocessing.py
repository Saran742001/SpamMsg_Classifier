import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (safe if already downloaded)
nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Cleans raw text for sentiment analysis
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    words = text.split()

    # Remove stopwords & apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)


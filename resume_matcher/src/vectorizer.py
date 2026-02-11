from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_texts(job_description, resumes):
    """
    Convert job description & resumes into TF-IDF vectors
    """
    documents = [job_description] + list(resumes.values())

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix, vectorizer

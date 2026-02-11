from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(tfidf_matrix, resumes):
    similarities = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    scores = {}
    for idx, resume_name in enumerate(resumes.keys()):
        scores[resume_name] = round(float(similarities[idx]), 3)

    return scores

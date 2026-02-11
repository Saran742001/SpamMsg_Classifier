import numpy as np

def extract_keywords(vectorizer, tfidf_matrix, top_n=15):
    """
    Extract top keywords from the job description (first row of tfidf_matrix)
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get the job description vector (first row)
    job_vector = tfidf_matrix[0]
    
    # Convert sparse matrix to dense array for sorting
    dense_vector = job_vector.toarray().flatten()
    
    # Get indices of top_n values (exclude 0 scores if possible, but taking top N is usually fine)
    top_indices = dense_vector.argsort()[-top_n:][::-1]
    
    return set(feature_names[top_indices])

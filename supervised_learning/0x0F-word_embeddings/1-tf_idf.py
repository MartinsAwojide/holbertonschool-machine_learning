#!/usr/bin/env python3
"""creates a TF-IDF embedding"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding
    Args:
        sentences: is a list of sentences to analyze
        vocab: is a list of the vocabulary words to use for the analysis

    Returns: embeddings, features
    - embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
    - s is the number of sentences in sentences
    - f is the number of features analyzed
    - features is a list of the features used for embeddings
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = X.toarray()
    return embeddings, features

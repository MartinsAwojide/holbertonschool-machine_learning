#!/usr/bin/env python3
"""multivariate uniform distribution"""
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means
    Args:
        X:is a numpy.ndarray of shape (n, d) containing the dataset that will
        be used for K-means clustering
        - n is the number of data points
        - d is the number of dimensions for each data point
        k: is a positive integer containing the number of clusters
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None

    _, d = X.shape
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    return np.random.uniform(min, max, size=(k, d))

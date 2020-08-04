#!/usr/bin/env python3
"""Performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        - n is the number of data points
        - d is the number of dimensions for each data point
        k: is a positive integer containing the number of clusters
        iterations:

    Returns: C, classes, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster
    classes is a numpy.ndarray of shape (n,) containing the index of the
    cluster in C that each data point belongs to
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None)
    if type(k) is not int or k <= 0:
        return (None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None)

    _, d = X.shape
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    # Place centroid randomly for each k
    centroids = np.random.uniform(min, max, size=(k, d))

    for i in range(iterations):
        copy = centroids.copy()
        # Calculate the nearest centroid for each point
        # broadcast rules: Two dimensions are compatible when
        # they are equal, or one of them is 1. Then operate in that axis
        distances = np.linalg.norm((X - centroids[:, np.newaxis]), axis=2)
        # show the nearest centroid where each point is, I called them class
        classes = distances.argmin(axis=0)
        for j in range(k):
            # randomly place centroid (class) if doesn't have a point within it
            if (X[classes == j].size == 0):
                centroids[j] = np.random.uniform(min, max, size=(1, d))
            # move centroid to the mean location
            else:
                centroids[j] = (X[classes == j].mean(axis=0))
        # update and then copy for comparison
        distances = np.linalg.norm((X - centroids[:, np.newaxis]), axis=2)
        classes = distances.argmin(axis=0)
        # if there is no more changes break
        if (copy == centroids).all():
            break

    return centroids, classes

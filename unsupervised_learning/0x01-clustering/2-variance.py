#!/usr/bin/env python3
"""calculates the total intra-cluster variance for a data set"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    Args:
        X:  is a numpy.ndarray of shape (n, d) containing the data set
        C: is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster

    Returns: var, or None on failure. var is the total variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.square(X - C[:, np.newaxis]).sum(axis=2)
    min_distances = np.min(distances, axis=0)
    var = np.sum(min_distances)

    return var

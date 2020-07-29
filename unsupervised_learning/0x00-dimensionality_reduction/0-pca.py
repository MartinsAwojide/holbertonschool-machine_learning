#!/usr/bin/env python3
"""Performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    Args:
        X: is a numpy.ndarray of shape (n, d) where
        - n is the number of data points
        - d is the number of dimensions in each point
        - all dimensions have a mean of 0 across all data points
        var: is the fraction of the variance that the PCA transformation
        should maintain

    Returns: the weights matrix, W, that maintains var fraction
    of X‘s original variance
    """
    u, s, vh = np.linalg.svd(X)
    cumulative = np.cumsum(s)
    threshold = cumulative[len(cumulative) - 1] * var
    mask = np.where(threshold > cumulative)
    var = cumulative[mask]
    idx = len(var) + 1
    W = vh.T
    Wr = W[:, 0:idx]
    return Wr

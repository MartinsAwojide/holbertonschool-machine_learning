#!/usr/bin/env python3
"""Calculates the expectation step"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the data set
        g: is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
    Returns: pi, m, S, or None, None, None on failure
    - pi is a numpy.ndarray of shape (k,) containing the updated priors
    for each cluster
    - m is a numpy.ndarray of shape (k, d) containing the updated centroid
    means for each cluster
    - S is a numpy.ndarray of shape (k, d, d) containing the updated
    covariance matrices for each cluster
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None

    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None

    if X.shape[0] != g.shape[1]:
        return None, None, None

    posterior = np.sum(g, axis=0)
    posterior = np.sum(posterior)
    if (int(posterior) != X.shape[0]):
        return (None, None, None)

    n, d = X.shape
    k, _ = g.shape

    # ∑i=1N ri * k
    N_soft = np.sum(g, axis=1)

    pi = N_soft / n

    mean = np.zeros((k, d))
    cov = np.zeros((k, d, d))
    for i in range(k):
        r = g[i].reshape(1, -1)
        den = N_soft[i]

        mean[i] = np.dot(r, X) / den

        first = r * (X - mean[i]).T
        cov[i] = np.dot(first, (X - mean[i])) / den

    return pi, mean, cov

#!/usr/bin/env python3
"""
Initializes all variables required to calculate the P affinities in t-SNE
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        to be transformed by t-SNE
        perplexity: is the perplexity that all Gaussian distributions
        should have

    Returns: (D, P, betas, H)
    D: a numpy.ndarray of shape (n, n) that calculates
    the pairwise distance between two data points
    P: a numpy.ndarray of shape (n, n) initialized to all 0‘s that
    will contain the P affinities
    betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s
    that will contain all of the beta values
    H: is the Shannon entropy for perplexity perplexity
    """
    (n, d) = X.shape
    # np.square is the elementwise square; axis=1 means sum the rows.
    # so sum_X is the vector of the norms of the x_i
    sum_X = np.sum(np.square(X), 1)
    # ||x_i-x_j||^2=||x_i||^2+||x_j||**2-2(x_i,x_j)
    # np.dot(X,X.T) has entries (x_i,x_j)
    # in position (i,j) you add ||x_i||^2 and ||x_j||^2 from the two sums
    # with the transpose it is added the x_j^2 missing for each ||x_i-x_j||^2
    # the result is that D is a symmetric matrix with the pairwise distances.
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    """ another version
    n, _ = X.shape
    # new axis to broadcast (1, 2500, 50) - (2500, 1, 50)
    # by broadcasting it operates each element
    # so the square of the substract is shape (2500, 2500, 50)
    X1 = X[np.newaxis, :, :]
    X2 = X[:, np.newaxis, :]
    # if operates in axis 2, it desapears and becomes (2500, 2500)
    D = np.sum(np.square(X1 - X2), axis=2)
    """

    np.fill_diagonal(D, 0)
    betas = np.ones((n, 1))
    P = np.zeros((n, n))
    H = np.log2(perplexity)

    return D, P, betas, H

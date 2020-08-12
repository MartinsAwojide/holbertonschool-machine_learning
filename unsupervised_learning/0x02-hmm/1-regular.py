#!/usr/bin/env python3
"""Regular Markov chain"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain
    Args:
        P: is a is a square 2D numpy.ndarray of shape (n, n) representing
        the transition matrix
        - P[i, j] is the probability of transitioning from state i to state j
        - n is the number of states in the markov chain

    Returns: a numpy.ndarray of shape (1, n) containing the steady
    state probabilities, or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n = P.shape[0]
    if n != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None

    _, eigen_vec = np.linalg.eig(P.T)
    # normalize and only real numbers
    normalize = eigen_vec / eigen_vec.sum().real
    aux = np.dot(normalize.T, P)

    for i in aux:
        if (i > 0).all() and np.isclose(i.sum(), 1):
            return i.reshape(1, n)

    return None

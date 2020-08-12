#!/usr/bin/env python3
"""Absorbing Markov chain"""
import numpy as np


def absorbing(P):
    """
    determines if a markov chain is absorbing
    Args:
        P: is a is a square 2D numpy.ndarray of shape (n, n)
        representing the transition matrix
        - P[i, j] is the probability of transitioning from state i to state j
        - n is the number of states in the markov chain

    Returns: True if it is absorbing, or False on failure

    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    if np.sum(P, axis=1).all() != 1:
        return None
    n = P.shape[0]
    if n != P.shape[1]:
        return False
    diag = np.diag(P)
    if (diag == 1).all():
        return True
    idx = np.where(diag == 1)
    if len(idx[0]) == 0:
        return False
    aux = diag == 1

    for i in range(n):
        for j in range(n):
            if P[i][j] > 0 and aux[j]:
                aux[i] = 1
    if aux.all() == 1:
        return True
    return False

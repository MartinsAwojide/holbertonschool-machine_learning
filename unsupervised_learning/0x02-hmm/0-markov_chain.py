#!/usr/bin/env python3
"""Markov chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular
    state after a specified number of iterations
    Args:
        P: is a square 2D numpy.ndarray of shape (n, n) representing
        the transition matrix
        - P[i, j] is the probability of transitioning from state i to state j
        - n is the number of states in the markov chain
        s: is a numpy.ndarray of shape (1, n) representing the probability
        of starting in each state
        t: is the number of iterations that the markov chain has been through

    Returns: a numpy.ndarray of shape (1, n) representing the probability
    of being in a specific state after t iterations, or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n = P.shape[0]
    if n != P.shape[1]:
        return None
    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    if type(t) != int or t < 0:
        return None
    m = s.shape[1]
    if n != m:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if np.sum(s) != 1:
        return None

    prob = s.copy()

    for i in range(t):
        prob = np.matmul(prob, P)

    return prob

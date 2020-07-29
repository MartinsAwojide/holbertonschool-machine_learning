#!/usr/bin/env python3
"""Calculates the Shannon entropy"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point
    Args:
        Di: is a numpy.ndarray of shape (n - 1,) containing the pariwise
        distances between a data point and all other points except itself
        - n is the number of data poin
        beta: is the beta value for the Gaussian distribution

    Returns: (Hi, Pi)
    Hi: the Shannon entropy of the points
    Pi: a numpy.ndarray of shape (n - 1,) containing
    the P affinities of the points
    """
    # ecuation of P(j|i)
    # beta = 1 / (2σ)^2

    num = np.exp(-Di.copy() * beta)
    # sumP is the denominator, the normalizing factor
    den = np.sum(np.exp(-Di.copy() * beta))
    Pi = num / den

    # ecuation of H(Pi)
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi

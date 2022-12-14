#!/usr/bin/env python3
"""Calculates the cost of the t-SNE transformation"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation
    Args:
        P: is a numpy.ndarray of shape (n, n) containing the P affinities
        Q: is a numpy.ndarray of shape (n, n) containing the Q affinities

    Returns: C, the cost of the transformation
    """
    # np.maxima It compares two arrays and returns a new
    # array containing the element-wise maxima
    # if Q or P 0 indef
    Q = np.maximum(Q, 1e-12)
    P = np.maximum(P, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C

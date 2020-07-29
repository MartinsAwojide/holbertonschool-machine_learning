#!/usr/bin/env python3
"""Calculates the Q affinities"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities
    Args:
        Y: is a numpy.ndarray of shape (n, ndim) containing
        the low dimensional transformation of X
        - n is the number of points
        - ndim is the new dimensional representation of X

    Returns: Q, num
    Q: is a numpy.ndarray of shape (n, n) containing the Q affinities
    num: is a numpy.ndarray of shape (n, n) containing the numerator
    of the Q affinities
    """
    sum_Y = np.sum(np.square(Y), 1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    # t-distribution
    # yi -yj, subs of each point against others
    num = (1 + D) ** (-1)
    # distance with itself = 0
    np.fill_diagonal(num, 0)
    # sum yk - yl, sum of all the distances num matrix
    den = np.sum(num)
    Q = num / den
    return Q, num

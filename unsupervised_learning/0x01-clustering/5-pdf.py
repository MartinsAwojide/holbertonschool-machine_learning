#!/usr/bin/env python3
"""Calculates the probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the data points
        whose PDF should be evaluated
        m: is a ndarray of shape (d,) containing the mean of the distribution
        S: is a numpy.ndarray of shape (d, d) containing the covariance
        of the distributio

    Returns: P, or None on failure
    P is a numpy.ndarray of shape (n,) containing the PDF values for each point
    All values in P have a minimum value of 1e-300
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    # Multivariate normal distribution
    # Non-degenerate case formula
    den = 1 / np.sqrt((2 * np.pi) ** d * det)
    part1 = np.matmul((-(X - m) / 2), inv)
    part2 = np.matmul(part1, (X - m).T).diagonal()
    num = np.exp(part2)

    pdf = num * den
    P = np.where(pdf < 1e-300, 1e-300, pdf)

    return P

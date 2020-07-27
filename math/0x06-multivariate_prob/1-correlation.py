#!/usr/bin/env python3
"""Calculates a correlation matrix"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    Args:
       C: is a numpy.ndarray of shape (d, d) containing a covariance matrix
       - d is the number of dimensions

    Returns:
    numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    d = C.shape[0]
    if type(C) != np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if C.shape != (d, d):
        raise ValueError("C must be a 2D square matrix")

    variance = np.sqrt(np.diag(C))
    # u = (u1, u2, ... um), v = (v1, v2, ... vn)
    # their outer product u ⊗ v is defined as the m × n matrix A
    # obtained by multiplying each element of u by each element of v
    outer_product = np.outer(variance, variance)
    correlation = C / outer_product

    return correlation

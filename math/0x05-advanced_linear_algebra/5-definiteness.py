#!/usr/bin/env python3
"""Calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
    Args:
        matrix: is a numpy.ndarray of shape (n, n) whose definiteness
        should be calculated

    Returns:the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite
    If matrix does not fit any of the above categories, return None
    """

    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix) < 2:
        return None
    row, col = matrix.shape

    if row != col:
        return None

    # Matrix has to be symmetric to calculate definiteness
    if not (matrix == matrix.T).all():
        return None

    eigen = np.linalg.eigvals(matrix)
    pos = 0
    neg = 0
    zero = 0
    size = len(eigen)
    for i in eigen:
        if i > 0:
            pos += 1
        if i < 0:
            neg += 1
        if i == 0:
            zero += 1

    if pos == size:
        return "Positive definite"
    elif neg == size:
        return "Negative definite"
    elif pos > 0 and neg > 0:
        return "Indefinite"
    elif pos > 0 and zero > 0:
        return "Positive semi-definite"
    elif neg > 0 and zero > 0:
        return "Negative semi-definite"
    else:
        return "Indefinite"

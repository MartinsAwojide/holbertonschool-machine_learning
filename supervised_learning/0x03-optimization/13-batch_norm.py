#!/usr/bin/env python3
"""batch norm"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an output of a neural network using batch normalization
    :param Z: ndarray of shape (m, n) that should be normalized
    :param gamma: is a numpy.ndarray of shape (1, n) containing the scales
    :param beta: is a numpy.ndarray of shape (1, n) containing the offsets
    :param epsilon: is a small number used to avoid division by zero
    :return: the normalized Z matrix
    """
    mean = np.mean(Z, axis=0)  # axis=0, Z.shape (m,n) formula: 1/m * np.sum(Z)
    stdv = np.std(Z, axis=0)
    Znorm = (Z - mean) / np.sqrt(stdv ** 2 + epsilon)
    return gamma * Znorm + beta

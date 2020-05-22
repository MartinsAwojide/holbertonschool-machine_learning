#!/usr/bin/env python3
"""Shuffle data"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way
    :param X:  is the first numpy.ndarray of shape (m, nx) to shuffle
            m is the number of data points
            nx is the number of features in X
    :param Y: is the second numpy.ndarray of shape (m, ny) to shuffle
            m is the same number of data points as in X
            ny is the number of features in Y

    :return: the shuffled X and Y matrices
    """
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    return shuffled_X, shuffled_Y

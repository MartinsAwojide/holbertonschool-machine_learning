#!/usr/bin/env python3
"""Multivariate Normal distribution"""
import numpy as np


class MultiNormal():
    """Multinormal class"""
    def __init__(self, data):
        """
        Class constructor
        Args:
            data: data is a numpy.ndarray of shape (d, n):
            - n is the number of data points
            - d is the number of dimensions in each data point
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")

        d, n = data.shape
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        num = data - self.mean
        self.cov = np.dot(num, num.T) / (n - 1)

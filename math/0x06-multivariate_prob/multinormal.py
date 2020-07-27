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
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        d, n = data.shape
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        num = data - self.mean
        self.cov = np.dot(num, num.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        Args:
            x: is a numpy.ndarray of shape (d, 1) containing
            the data point whose PDF should be calculated

        Returns: the value of the P
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        part1 = 1 / ((2 * np.pi)**(d/2) * np.sqrt(det))
        part2 = np.dot((x-self.mean).T, inv)
        part3 = np.dot(part2, (x - self.mean) / -2)
        pdf = part1 * np.exp(part3)

        return pdf[0][0]

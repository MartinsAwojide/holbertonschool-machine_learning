#!/usr/bin/env python3
"""Calculates the likelihood of obtaining this data"""
import numpy as np


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data
    Args:
        x: is the number of patients that develop severe side effects
        n: is the total number of patients observed
        P: is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
    Returns:
    a 1D numpy.ndarray containing the likelihood of obtaining the data,
    x and n, for each probability in P, respectively
    """
    if not(n > 0 and type(n) is int):
        raise ValueError("n must be a positive integer")
    if not(n >= 0 and type(n) is int):
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) != np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for i in P:
        if not (i >= 0 and i <= 1):
            msg = "All values in P must be in the range [0, 1]"
            raise ValueError(msg)

    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_dif = np.math.factorial(n-x)
    like = P ** x * (fact_n / (fact_x * fact_dif)) * (1-P) ** (n-x)
    return like

#!/usr/bin/env python3
"""Calculates the posterior probability of a uniform distribution"""
from scipy import math, special, stats
import numpy as np


def posterior(x, n, p1, p2):
    """
    calculates the likelihood of obtaining this data
    Args:
        x: is the number of patients that develop severe side effects
        n: is the total number of patients observed
        p1: is the lower bound on the range
        p2: is the upper bound on the range
    Returns:
    the posterior probability that p is within the range [p1, p2] given x and n
    """
    if not (type(n) is int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not (type(x) is int) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or not 0 <= p1 <= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if type(p2) is not float or not 0 <= p2 <= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    P = (x * (p1 + p2)) / (n * (p1 + p2))
    num = (special.factorial(n) / (special.factorial(x) *
                                   special.factorial(n - x))) * (P ** x) * \
          ((1 - P) ** (n - x))

    Pr = 1
    intersection = num * Pr

    pos = intersection / np.sum(intersection)

    return pos

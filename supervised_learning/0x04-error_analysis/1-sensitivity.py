#!/usr/bin/env python3
"""Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    :param confusion: is a confusion numpy.ndarray of shape (classes, classes)
    row represent the correct labels and column represent the predicted labels
    :return: ndarray shape (classes,) containing the sensitivity of each class
    """
    true_positive = np.diag(confusion)  # True Positives are diagonal elements
    positives = np.sum(confusion, axis=1)  # P = TP + FN
    return true_positive / positives  # TP / P

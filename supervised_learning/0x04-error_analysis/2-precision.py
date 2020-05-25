#!/usr/bin/env python3
"""Precision"""
import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix
    :param confusion: is a confusion numpy.ndarray of shape (classes, classes)
    row represent the correct labels and column represent the predicted labels
    :return: ndarray shape (classes,) containing the sensitivity of each class
    """
    true_positive = np.diag(confusion)  # True Positives are diagonal elements
    false_positive = confusion.sum(axis=0) - np.diag(confusion)
    return true_positive / (true_positive + false_positive)  # TP / (TP + FP)

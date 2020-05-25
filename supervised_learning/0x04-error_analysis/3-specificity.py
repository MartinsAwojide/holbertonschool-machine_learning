#!/usr/bin/env python3
"""Specificity"""
import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix
    :param confusion: is a confusion numpy.ndarray of shape (classes, classes)
    row represent the correct labels and column represent the predicted labels
    :return: ndarray shape (classes,) containing the sensitivity of each class
    """
    TP = np.diag(confusion)  # True Positives are diagonal elements
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    true_negative = confusion.sum() - (FP + FN + TP)
    return true_negative / (true_negative + FP)

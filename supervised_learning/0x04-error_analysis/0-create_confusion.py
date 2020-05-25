#!/usr/bin/env python3
"""Creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    :param labels: is a one-hot numpy.ndarray of shape (m, classes)
    containing the correct labels for each data point
        m is the number of data points
        classes is the number of classes
    :param logits:  is a one-hot numpy.ndarray of shape (m, classes)
    containing the predicted labels
    :return: confusion ndarray of shape (classes, classes) with row indices
    representing correct labels and column indices representing predicted label
    """
    # m, classes = labels.shape  # label shape (m, classes)
    # result = np.zeros((classes, classes))  # shape (classes, classes)
    # predicted = np.argmax(logits, axis=1)  # label == 1
    # actual = np.argmax(labels, axis=1)
    # for a, p in zip(actual, predicted):
    #    result[a][p] += 1
    return np.matmul(labels.T, logits)

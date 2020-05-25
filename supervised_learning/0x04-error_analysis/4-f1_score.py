#!/usr/bin/env python3
"""f1 score"""
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix
    :param confusion: is a confusion numpy.ndarray of shape (classes, classes)
    row represent the correct labels and column represent the predicted labels
    :return: ndarray shape (classes,) containing the sensitivity of each class
    """
    PPV = precision(confusion)
    TPR = sensitivity(confusion)
    return 2 * ((PPV * TPR) / (PPV + TPR))

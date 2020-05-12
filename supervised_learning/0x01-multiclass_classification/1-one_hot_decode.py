#!/usr/bin/env python3
"""One hot decode"""
import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels
    :param one_hot: is a one-hot encoded numpy.ndarray with shape (classes, m)
                    classes is the maximum number of classes
                    m is the number of examples
    :return:  a numpy.ndarray with shape (m, ) containing the numeric labels
    for each example, or None on failure
    """
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot.shape) != 2 or len(one_hot) == 0:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None

    return np.argmax(one_hot, axis=0)

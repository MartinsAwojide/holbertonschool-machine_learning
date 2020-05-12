#!/usr/bin/env python3
"""One hot encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    :param Y: is a ndarray with shape (m,) containing numeric class labels
              m is the number of examples
    :param classes:  is the maximum number of classes found in Y
    :return:
    """
    if type(classes) is not int or classes <= max(Y):
        return None
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    encode = (np.identity(classes)[Y])  # use the identity array where is Y
    # np identity is a special case of np eye
    return encode.T

#!/usr/bin/env python3
"""one_hot encode with keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix
    :param labels: class vector to be converted into a matrix
    :param classes: the last dimension of the one-hot matrix must be
    the number of classes
    :return: the one-hot matrix
    """
    one_hot_encode = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot_encode

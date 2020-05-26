#!/usr/bin/env python3
"""L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization. The neural network uses tanh activations on each
    layer except the last, which uses a softmax activation
    :param Y: is a one-hot numpy.ndarray of shape (classes, m) that contains
    the correct labels for the data
            classes is the number of classes
            m is the number of data points
    :param weights: dictionary of the weights and biases of the neural network
    :param cache: dictionary of the outputs of each layer of the neural network
    :param alpha: is the learning rate
    :param lambtha: is the L2 regularization parameter
    :param L: is the number of layers of the network
    :return: The weights and biases of the network should be updated in place
    """
    for i in reversed(range(L)):
        key_w = "W{}".format(i + 1)
        key_b = "b{}".format(i + 1)
        key_a = "A{}".format(i + 1)
        A = cache[key_a]
        m = Y.shape[1]
        if i == L - 1:
            dz = A - Y
            w = weights[key_w]
        else:
            g = 1 - (A * A)
            part1 = np.matmul(w.T, dz)
            dz = part1 * g
            w = weights[key_w]
        dw = np.matmul(cache["A{}".format(i)], dz.T) / m + (lambtha * w.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[key_w] = weights[key_w] - (alpha * dw.T)
        weights[key_b] = weights[key_b] - (alpha * db)

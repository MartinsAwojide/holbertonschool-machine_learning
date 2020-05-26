#!/usr/bin/env python3
"""Dropout regularization"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent
    :param Y: Y is a one-hot numpy.ndarray of shape (classes, m)
    that contains the correct labels for the data
                classes is the number of classes
                m is the number of data points
    :param weights: dictionary of the weights and biases of the neural network
    :param cache: dictionary of the outputs and dropout masks of each layer
    :param alpha: is the learning rate
    :param keep_prob: is the probability that a node will be kept
    :param L: is the number of layers of the network
    :return: The weights of the network should be updated in place
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
            # apply mask if 0=0, if 1=tanh. The derivative is (1-A^2)*D + A*0
            dz = part1 * g * cache["D{}".format(i + 1)]
            dz = dz / keep_prob
            w = weights[key_w]
        dw = np.matmul(cache["A{}".format(i)], dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[key_w] = weights[key_w] - (alpha * dw.T)
        weights[key_b] = weights[key_b] - (alpha * db)

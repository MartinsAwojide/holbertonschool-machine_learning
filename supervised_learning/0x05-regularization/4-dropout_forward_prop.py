#!/usr/bin/env python3
"""Dropout regularization"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that conducts forward propagation using Dropout
    :param X: ndarray of shape (nx, m) containing input data for the network
                nx is the number of input features
                m is the number of data points
    :param weights: dictionary of the weights and biases of the neural network
    :param L: the number of layers in the network
    :param keep_prob: the probability that a node will be kept
    :return: dictionary containing the outputs of each layer and
    the dropout mask used on each layer
    """
    cache = {'A0': X}
    for i in range(L):
        key_w = "W{}".format(i + 1)
        key_b = "b{}".format(i + 1)
        key_a = "A{}".format(i)
        new_key_a = "A{}".format(i + 1)
        d_key = "D{}".format(i + 1)
        z = np.matmul(weights[key_w], cache[key_a]) + weights[key_b]
        if i == L - 1:
            # Softmax
            t = np.exp(z)
            activation = np.exp(z) / t.sum(axis=0, keepdims=True)
        else:
            # Tanh
            activation = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            random = np.random.rand(activation.shape[0], activation.shape[1])
            d = (random < keep_prob).astype(int)
            activation = np.multiply(activation, d)
            activation = activation / keep_prob
            cache[d_key] = d
        cache[new_key_a] = activation
    return cache

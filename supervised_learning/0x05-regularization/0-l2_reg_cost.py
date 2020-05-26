#!/usr/bin/env python3
"""L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    :param cost: is the cost of the network without L2 regularization
    :param lambtha: is the regularization parameter
    :param weights: is a dictionary of the weights and biases (numpy.ndarrays)
    of the neural network
    :param L: is the number of layers in the neural network
    :param m: is the number of data points used
    :return: the cost of the network accounting for L2 regularization
    """
    get_weights = np.zeros(L)
    # calculate frobenius norm for each W
    for i in range(L):
        get_weights[i] = np.linalg.norm(weights['W' + str(i+1)])
    L2_cost = np.sum(get_weights) * lambtha / (m * 2)
    cost = cost + L2_cost
    return cost

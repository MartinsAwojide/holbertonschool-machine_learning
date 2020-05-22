#!/usr/bin/env python3
"""variables RMSprop"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm
    :param alpha: is the learning rate
    :param beta2: is the RMSProp weight
    :param epsilon: is a small number to avoid division by zero
    :param var: ndarray containing the variable to be updated
    :param grad: ndarray containing the gradient of var
    :param s: is the previous second moment of var
    :return: the updated variable and the new moment, respectively
    """
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - (alpha * grad) / (np.sqrt(s) + epsilon)
    return var, s

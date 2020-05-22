#!/usr/bin/env python3
"""variables Adam"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using the Adam optimization algorithm
    :param alpha: is the learning rate
    :param beta1: is the weight used for the first moment
    :param beta2: is the weight used for the second moment
    :param epsilon: is a small number to avoid division by zero
    :param var: ndarray containing the variable to be updated
    :param grad: ndarray containing the gradient of var
    :param v: is the previous first moment of var
    :param s: is the previous second moment of var
    :param t: is the time step used for bias correction
    :return: updated variable, the new first moment, and the new second moment
    """
    v = beta1 * v + (1 - beta1) * grad
    v_corrected = v / (1 - beta1 ** t)

    s = beta2 * s + (1 - beta2) * grad ** 2
    s_corrected = s / (1 - beta2 ** t)
    var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))
    return var, v, s

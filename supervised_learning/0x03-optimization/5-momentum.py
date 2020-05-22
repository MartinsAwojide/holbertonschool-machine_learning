#!/usr/bin/env python3
"""update variable momentum"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent with momentum optimization
    :param alpha: is the learning rate
    :param beta1: is the momentum weight
    :param var: ndarray containing the variable to be updated
    :param grad: ndarray containing the gradient of var
    :param v: is the previous first moment of var
    :return: the updated variable and the new moment, respectively
    """
    variable = beta1 * v + (1 - beta1) * grad
    new_variable = var - (alpha * variable)
    return new_variable, variable

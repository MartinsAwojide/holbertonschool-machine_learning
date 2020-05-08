#!/usr/bin/env python3
"""Class NeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        """
        class constructor
        :param nx: is the number of input features
        :param layers: is a list representing the number of nodes in each
        layer of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        layers_num = np.array(layers)
        if np.any(layers_num < 1):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            key_w = "W{}".format(i+1)
            key_b = "b{}".format(i+1)
            if i == 0:
                weight = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.weights[key_w] = weight
            else:
                weight = np.random.randn(layers[i], layers[i - 1]) * \
                         np.sqrt(2 / layers[i - 1])
                self.weights[key_w] = weight
            bias = np.zeros((layers[i], 1))
            self.weights[key_b] = bias

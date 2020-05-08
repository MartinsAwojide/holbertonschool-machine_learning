#!/usr/bin/env python3
"""Class NeuralNetwork"""
import numpy as np


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
        if np.any(layers_num < 1) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            key_w = "W{}".format(i+1)
            key_b = "b{}".format(i+1)
            if i == 0:
                weight = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights[key_w] = weight
            else:
                weight = np.random.randn(layers[i], layers[i - 1]) * \
                         np.sqrt(2 / layers[i - 1])
                self.__weights[key_w] = weight
            bias = np.zeros((layers[i], 1))
            self.__weights[key_b] = bias

    @property
    def L(self):
        """
        A attribute getter
        :return: The number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """ cache attribute getter.
        :return: A dictionary to hold all intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """ weights attribute getter.
        :return: A dictionary to hold all weights and biased of the network.
        """
        return self.__weights

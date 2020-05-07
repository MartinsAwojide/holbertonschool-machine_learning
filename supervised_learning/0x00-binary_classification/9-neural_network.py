#!/usr/bin/env python3
"""Class NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """ defines a neural network with one hidden layer performing
    binary classification"""
    def __init__(self, nx, nodes):
        """
        class constructor
        :param nx: is the number of input features
        :param nodes: is the number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        W1 attribute getter.
        :return: The weights vector for the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        b1 attribute getter.
        :return: The bias for the hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """
        A1 attribute getter.
        :return: The Activation output for the inner layer
        """
        return self.__A1

    @property
    def W2(self):
        """
        W1 attribute getter.
        :return: The weights vector for the output neuron
        """
        return self.__W2

    @property
    def b2(self):
        """
        b2 attribute getter.
        :return: The bias for the output neuron.
        """
        return self.__b2

    @property
    def A2(self):
        """
        A2 attribute getter.
        :return: The activated output for the output neuron (prediction)
        """
        return self.__A2

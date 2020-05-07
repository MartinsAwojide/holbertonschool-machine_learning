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

    def forward_prop(self, X):
        """
        defines a neural network with one hidden layer performing
        binary classification
        :param X: is a ndarray with shape (nx, m) that contains the input data
                  nx is the number of input features to the neuron
                  m is the number of examples
        :return: the private attributes __A1 and __A2, respectively
        """
        z = np.matmul(self.__W1, X) + self.__b1
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A1 = sigmoid
        z = np.matmul(self.__W2, self.A1) + self.__b2
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A2 = sigmoid
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        :param Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :param A: is a numpy.ndarray with shape (1, m) containing
        the activated output of the neuron for each example
        :return: the cost
        """
        summatory = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        constant = -(1/A.shape[1])
        return constant * summatory.sum()

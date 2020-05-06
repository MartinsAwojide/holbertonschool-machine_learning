#!/usr/bin/env python3
"""Class Neuron"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """
        class constructor
        :param nx: is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        :return:The weights vector for the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        :return:The bias for the neuron
        """
        return self.__b

    @property
    def A(self):
        """
        :return:The activated output of the neuron (prediction)
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        :param X: is a numpy.ndarray with shape (nx, m)
        that contains the input data
        :return: the private attribute __A
        """
        z = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A = sigmoid
        return self.__A

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

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        :param X: numpy.ndarray with shape (nx, m) that contains the input data
        :param Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :return: the neuron’s prediction and the cost of the network
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        self.__A = np.where(self.__A >= 0.5, 1, 0)
        return self.__A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        :param X: is a ndarray with shape (nx, m) that contains the input data
                  nx is the number of input features to the neuron
                  m is the number of examples
        :param Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :param A: is a numpy.ndarray with shape (1, m) containing the
        activated output of the neuron for each example
        :param alpha: is the learning rate
        :return: Updates the private attributes __W and __b
        """
        self.forward_prop(X)
        dz = A - Y
        m = Y.shape[1]
        dw = np.matmul(X, dz.T) / m  # A, Y are transpose for operation
        db = dz.sum() / m
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron by updating the private attributes __W, __b, and __A
        :param X: is a ndarray with shape (nx, m) that contains the input data
                  nx is the number of input features to the neuron
                  m is the number of examples
        :param Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :param iterations: is the number of iterations to train over
        :param alpha: is the learning rate
        :param verbose: is a boolean that defines whether or not to print
        information about the training.
        :param graph:  is a boolean that defines whether or not to graph
        information about the training once the training has completed
        :param step: step iterations
        :return: the evaluation of the training data after iterations of
        training have occurred
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True and graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        graph_iteration = []
        graph_cost = []

        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A)

            if step and (i % step == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))
                graph_iteration.append(i)
                graph_cost.append(cost)

            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)

        if graph is True:
            plt.plot(graph_iteration, graph_cost)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)

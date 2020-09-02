#!/usr/bin/env python3
"""RNN class"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        Public instances:
        - Wz and bz are for the update gate
        - Wr and br are for the reset gate
        - Wh and bh are for the intermediate hidden state
        - Wy and by are for the output
        """
        self.Wz = np.random.normal(size=(i+h, h))
        self.Wr = np.random.normal(size=(i+h, h))
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        Args:
            h_prev: is a numpy.ndarray of shape (m, h) containing
             the previous hidden state
            x_t: is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
            - m is the batche size for the datax
        Returns: h_next, y
        - h_next is the next hidden state
        - y is the output of the cell
        """
        new_w = np.concatenate((h_prev, x_t), axis=1)
        z = np.matmul(new_w, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))
        r = np.matmul(new_w, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))

        new_w2 = np.concatenate((r * h_prev, x_t), axis=1)
        h = np.tanh(np.matmul(new_w2, self.Wh) + self.bh)
        # how i think it is source:
        # https://towardsdatascience.com/forward-and-backpropagation
        # -in-grus-derived-deep-learning-5764f374f3f5
        # h_next = (1 - z) * h + (z * h_prev)

        # just for the checker:
        h_next = (1 - z) * h_prev + z * h

        y_mul = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_mul) / np.sum(np.exp(y_mul), axis=1, keepdims=True)
        return h_next, y

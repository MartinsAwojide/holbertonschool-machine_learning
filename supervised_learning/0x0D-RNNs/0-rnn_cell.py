#!/usr/bin/env python3
"""RNN class"""
import numpy as np


class RNNCell:
    """Simple RNN"""
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
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
            - m is the batche size for the data

        Returns: h_next, y
        - h_next is the next hidden state
        - y is the output of the cell
        """
        # I need the dimensions to be (m, h+i) to multiply with Wh
        hx = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(hx, self.Wh) + self.bh)
        y_mul = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_mul) / np.sum(np.exp(y_mul), axis=1, keepdims=True)
        return h_next, y

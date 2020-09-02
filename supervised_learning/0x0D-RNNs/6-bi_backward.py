#!/usr/bin/env python3
"""RNN class"""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        Public instances:
        - Whf and bhf are for the hidden states in the forward direction
        - Whb and bhb are for the hidden states in the backward direction
        - Wy and by are for the outputs
        """
        self.Whf = np.random.normal(size=(i+h, h))
        self.Whb = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(2*h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step
        Args:
            h_prev: is a numpy.ndarray of shape (m, h) containing
            the previous hidden state
            x_t: is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
            - m is the batch size for the data

        Returns: h_next, the next hidden state
        """
        new_w = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(new_w, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step
        Args:
            h_next: is a numpy.ndarray of shape (m, h) containing
            the next hidden state
            x_t: is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
            - m is the batch size for the data

        Returns: h_pev, the previous hidden state
        """
        new_w = np.concatenate((h_next, x_t), axis=1)
        h_pev = np.tanh(np.matmul(new_w, self.Whb) + self.bhb)

        return h_pev

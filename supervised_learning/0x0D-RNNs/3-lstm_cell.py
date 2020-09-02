#!/usr/bin/env python3
"""RNN class"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        Public instances:
        - Wf and bf are for the forget gate
        - Wu and bu are for the update gate
        - Wc and bc are for the intermediate cell state
        - Wo and bo are for the output gate
        - Wy and by are for the outputs
        """
        self.Wf = np.random.normal(size=(i+h, h))
        self.Wu = np.random.normal(size=(i+h, h))
        self.Wc = np.random.normal(size=(i+h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
        Args:
            h_prev: is a numpy.ndarray of shape (m, h) containing
             the previous hidden state
            - c_prev: is a numpy.ndarray of shape (m, h) containing
            the previous cell state
            x_t: is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
            - m is the batche size for the datax
        Returns: h_next, c_next, y
        - h_next is the next hidden state
        - c_next is the next cell state
        - y is the output of the cell
        """
        new_w = np.concatenate((h_prev, x_t), axis=1)
        f = np.matmul(new_w, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f))
        u = np.matmul(new_w, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))
        o = np.matmul(new_w, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))

        c_hat = np.tanh(np.matmul(new_w, self.Wc) + self.bc)
        c_next = u * c_hat + f * c_prev
        h_next = o * np.tanh(c_next)

        y_mul = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_mul) / np.sum(np.exp(y_mul), axis=1, keepdims=True)
        return h_next, c_next, y

#!/usr/bin/env python3
"""RNN forward"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    Args:
        X: is the data to be used, given as a numpy.ndarray of shape
        (t, m, i)
        - t is the maximum number of time steps
        - m is the batch size
        - i is the dimensionality of the data
        h_0:  is the initial hidden state, given as a numpy.ndarray
        of shape (m, h)
        - h is the dimensionality of the hidden state
    Returns: H, Y
    - H is a numpy.ndarray containing all of the hidden states
    - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    o = rnn_cell.Wy.shape[-1]
    h_prev = h_0
    H = np.zeros((t+1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    for i in range(t):
        x_t = X[i]
        h_prev, y_prev = rnn_cell.forward(h_prev, x_t)
        H[i+1] = h_prev
        Y[i] = y_prev
    return H, Y

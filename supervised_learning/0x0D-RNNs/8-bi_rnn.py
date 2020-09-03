#!/usr/bin/env python3
"""RNN class"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN
    Args:
        bi_cell:
        X: is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        - t is the maximum number of time steps
        - m is the batch size
        - i is the dimensionality of the data
        h_0: is the initial hidden state in the forward direction,
        given as a numpy.ndarray of shape (m, h)
        - h is the dimensionality of the hidden state
        h_t: is the initial hidden state in the backward direction,
        given as a numpy.ndarray of shape (m, h)

    Returns: H, Y
    H is a numpy.ndarray containing all of the concatenated hidden states
    Y is a numpy.ndarray containing all of the outputs

    """
    t, m, i = X.shape
    h = h_0.shape[1]
    h_for = np.zeros((t, m, h))
    h_back = np.zeros((t, m, h))
    h_ft = h_0
    h_bt = h_t
    for i in range(t):
        x_ft = X[i]
        x_bt = X[-(i+1)]
        # auxiliar to take the forward and backward
        h_ft = bi_cell.forward(h_ft, x_ft)
        h_bt = bi_cell.backward(h_bt, x_bt)
        # append in a numpy array
        h_for[i] = h_ft
        h_back[-(i+1)] = h_bt
    # concatenates the forward and backward into H
    H = np.concatenate((h_for, h_back), axis=-1)
    Y = bi_cell.output(H)
    return H, Y

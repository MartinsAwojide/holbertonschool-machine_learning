#!/usr/bin/env python3
"""forward propagation"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional layer of a neural network
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W:  is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        the kernels for the convolution
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output
        b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the
        biases applied to the convolution
        activation: is an activation function applied to the convolution
        padding: is a string that is either same or valid, indicating
        the type of padding used
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh is the stride for the height
            sw is the stride for the width

    Returns:
            the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    pad_h, pad_w = (0, 0)
    if padding == 'same':
        pad_h = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pad_w = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    pad_image = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                                (0, 0)), 'constant', constant_values=0)
    conv_h = int((h_prev + (2 * pad_h) - kh) / sh) + 1
    conv_w = int((w_prev + (2 * pad_w) - kw) / sw) + 1

    conv = np.zeros((m, conv_h, conv_w, c_new))

    for h in range(conv_h):
        for w in range(conv_w):
            for c in range(c_new):
                st_h = h * sh
                en_h = h * sh + kh
                st_w = w * sw
                en_w = w * sw + kw
                slide = pad_image[:, st_h:en_h, st_w:en_w]
                filter = W[:, :, :, c]
                aux = filter * slide
                conv[:, h, w, c] = aux.sum(axis=(1, 2, 3))
    Z = conv + b

    return activation(Z)

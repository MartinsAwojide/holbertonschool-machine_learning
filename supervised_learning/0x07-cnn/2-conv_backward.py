#!/usr/bin/env python3
"""backward propagation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network
    Args:
        dZ: dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
        the partial derivatives with respect to the unactivated output of
        the convolutional layer
            - m is the number of examples
            - h_new is the height of the output
            - w_new is the width of the output
            - c_new is the number of channels in the output
        A_prev:  is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
            - c_prev is the number of channels in the previous layer
        W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        the kernels for the convolution
            - kh is the filter height
            - kw is the filter width
        b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
        padding: is a string that is either same or valid, indicating the type
        of padding used
        stride: tuple of (sh, sw) containing the strides for the convolution
            - sh is the stride for the height
            - sw is the stride for the width

    Returns: the partial derivatives with respect to the previous
    layer (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    pad_h, pad_w = (0, 0)
    if padding == 'same':
        pad_h = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pad_w = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))

    # initialize the derivatives

    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # padding
    A_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                            (0, 0)), 'constant', constant_values=0)
    dA_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                             (0, 0)), 'constant', constant_values=0)

    dA_prev = np.zeros_like(A_pad)
    for i in range(m):
        A_pad_prev = A_pad[i]
        dA_pad_prev = dA_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    st_h = h * sh
                    en_h = h * sh + kh
                    st_w = w * sw
                    en_w = w * sw + kw
                    X = A_pad_prev[st_h:en_h, st_w:en_w, :]
                    dA_pad_prev[st_h:en_h, st_w:en_w] += \
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += X * dZ[i, h, w, c]

        if padding == 'valid':
            dA_prev[i, :, :, :] += dA_pad_prev
        elif padding == 'same':
            # if pad = 1: star at 1 and end -1, only takes inside matrix
            dA_prev[i, :, :, :] += dA_pad_prev[:, pad_h:-pad_h, pad_w:-pad_w]
    return dA_prev, dW, db

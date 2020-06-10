#!/usr/bin/env python3
"""backward propagation"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network
    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the output of the pooling layer
            - m is the number of examples
            - h_new is the height of the output
            - w_new is the width of the output
            - c is the number of channels
        A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c)
        containing the output of the previous layer
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
        kernel_shape: is a tuple of (kh, kw) containing the size of the kernel
            - kh is the kernel height
            - kw is the kernel width
        stride:  is a tuple of (sh, sw) containing the strides for the pooling
            - sh is the stride for the height
            - sw is the stride for the width
        mode: is a string containing either max or avg, indicating whether
        to perform maximum or average pooling, respectively

    Returns:
        the partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # initialize the derivatives
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    st_h = h * sh
                    en_h = h * sh + kh
                    st_w = w * sw
                    en_w = w * sw + kw

                    if mode == 'max':
                        # mask with 1 on max and 0 otherwise
                        X = A_prev[i, st_h:en_h, st_w:en_w, c]
                        mask = (np.max(X) == X).astype(int)
                        # da is prev da mul by mask to generate more dimensions
                        # than the da.
                        dA_prev[i, st_h:en_h, st_w:en_w, c] += \
                            dA[i, h, w, c] * mask
                    elif mode == 'avg':
                        # if 4 number mean = 3, 3*1 added 4 times is 12 / 4 = 3
                        average = dA[i, h, w, c] / (kh * kw)
                        one = np.ones(kernel_shape) * average
                        dA_prev[i, st_h:en_h, st_w:en_w, c] += one
    return dA_prev

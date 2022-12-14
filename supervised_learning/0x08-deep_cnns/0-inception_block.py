#!/usr/bin/env python3
"""Builds an inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block
    Args:
        A_prev: is the output from the previous layer
        filters: is a tuple or list containing:
                - F1 is the number of filters in the 1x1 convolution
                - F3R is the number of filters in the 1x1 convolution
                    before the 3x3 convolution
                - F3 is the number of filters in the 3x3 convolution
                - F5R is the number of filters in the 1x1 convolution before
                    the 5x5 convolution
                - F5 is the number of filters in the 5x5 convolution
                - FPP is the number of filters in the 1x1 convolution after
                    the max pooling
    Returns:
            the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal()

    conv_F1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                              kernel_initializer=init,
                              activation='relu')(A_prev)
    conv_F3R = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                               kernel_initializer=init,
                               activation='relu')(A_prev)
    conv_F3 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                              kernel_initializer=init,
                              activation='relu')(conv_F3R)
    conv_F5R = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                               kernel_initializer=init,
                               activation='relu')(A_prev)
    conv_F5 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                              kernel_initializer=init,
                              activation='relu')(conv_F5R)
    max_pool = K.layers.MaxPool2D(pool_size=3, strides=1,
                                  padding='same')(A_prev)
    conv_FPP = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same',
                               kernel_initializer=init,
                               activation='relu')(max_pool)
    concat = K.layers.concatenate([conv_F1, conv_F3, conv_F5, conv_FPP])
    return concat

#!/usr/bin/env python3
"""Projection block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds an projection block for ResNet
    Args:
        A_prev: is the output from the previous layer
        filters: is a tuple or list containing F11, F3, F12, respectively:
                - F11 is the number of filters in the first 1x1 convolution
                - F3 is the number of filters in the 3x3 convolution
                - F12 is the number of filters in the second 1x1 convolution
        s: is the stride of the first convolution in both the main path and
        the shortcut connection
    Returns: the activated output of the identity block
    """
    X_shortcut = A_prev
    init = K.initializers.he_normal()
    F11, F3, F12 = filters
    X = K.layers.Conv2D(filters=F11, kernel_size=1, kernel_initializer=init,
                        padding='same', strides=s)(A_prev)
    #  along the channels axis. A_prev.shape=(None, 224, 224, 256)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=F3, kernel_size=3, kernel_initializer=init,
                        padding='same', strides=(1, 1))(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=F12, kernel_size=1, kernel_initializer=init,
                        padding='same', strides=(1, 1))(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X_shortcut = K.layers.Conv2D(filters=F12, kernel_size=1,
                                 kernel_initializer=init,
                                 padding='same', strides=s)(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X

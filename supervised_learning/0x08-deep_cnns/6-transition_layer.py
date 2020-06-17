#!/usr/bin/env python3
"""Transition layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer
    Args:
        X: is the output from the previous layer
        nb_filters: is an integer representing the number of filters in X
        compression: is the compression factor for the transition layer
    The code implement compression as used in DenseNet-C
    All weights use he normal initialization
    All convolutions are preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively
    """
    init = K.initializers.he_normal()
    filter_compr = int(compression * nb_filters)

    norm = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation('relu')(norm)
    conv = K.layers.Conv2D(filters=filter_compr,
                           kernel_size=1,
                           padding='same',
                           kernel_initializer=init,
                           strides=(1, 1))(act)
    avg = K.layers.AveragePooling2D(pool_size=2,
                                    padding='same',
                                    strides=2)(conv)
    return avg, filter_compr

#!/usr/bin/env python3
"""Dense block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block
    Args:
        X: is the output from the previous layer
        nb_filters: is an integer representing the number of filters in X
        growth_rate: is the growth rate for the dense block
        layers: is the number of layers in the dense block
    You use the bottleneck layers used for DenseNet-B
    All weights use he normal initialization
    All convolutions are preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively
    """
    init = K.initializers.he_normal()

    for i in range(layers):
        norm1 = K.layers.BatchNormalization(axis=3)(X)
        act1 = K.layers.Activation('relu')(norm1)
        # k is growth rate. bottle neck = 4*k
        b_neck = K.layers.Conv2D(filters=4*growth_rate,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=init,
                                 strides=(1, 1))(act1)
        norm2 = K.layers.BatchNormalization(axis=3)(b_neck)
        act2 = K.layers.Activation('relu')(norm2)
        conv = K.layers.Conv2D(filters=growth_rate,
                               kernel_size=3,
                               padding='same',
                               kernel_initializer=init,
                               strides=(1, 1))(act2)
        X = K.layers.concatenate([X, conv])
        nb_filters += growth_rate
    return X, nb_filters

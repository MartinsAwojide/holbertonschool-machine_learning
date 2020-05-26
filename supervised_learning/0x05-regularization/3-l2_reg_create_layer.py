#!/usr/bin/env python3
"""L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization
    :param prev: is a tensor containing the output of the previous layer
    :param n: is the number of nodes the new layer should contain
    :param activation: the activation function that should be used on the layer
    :param lambtha: is the L2 regularization parameter
    :return: the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regulizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regulizer, name="layer")
    return layer(prev)

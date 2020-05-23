#!/usr/bin/env python3
"""batch norm"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow
    :param prev: is the activated output of the previous layer
    :param n: is the number of nodes in the layer to be created
    :param activation: is the activation function to be used on the output
    :return: a tensor of the activated output for the layer
    """
    # Layers
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = output(prev)

    # Gamma and Beta initialization
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]),
                       name="beta")

    # Batch normalization
    mean, var = tf.nn.moments(Z, axes=0)  # return tensor mean, variance
    b_norm = tf.nn.batch_normalization(Z, mean, var, offset=beta, scale=gamma,
                                       variance_epsilon=1e-8)
    if activation is None:
        return b_norm
    else:
        return activation(b_norm)

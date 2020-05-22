#!/usr/bin/env python3
"""RMSprop train"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm
    :param loss: is the loss of the network
    :param alpha: is the learning rate
    :param beta2: is the RMSProp weight
    :param epsilon: is a small number to avoid division by zero
    :return: the RMSProp optimization operation
    """
    return tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                     epsilon=epsilon).minimize(loss)

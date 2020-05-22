#!/usr/bin/env python3
"""Adam train"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    :param loss: is the loss of the network
    :param alpha: is the learning rate
    :param beta1: is the weight used for the first moment
    :param beta2: is the weight used for the second moment
    :param epsilon: is a small number to avoid division by zero
    :return: the Adam optimization operation
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)

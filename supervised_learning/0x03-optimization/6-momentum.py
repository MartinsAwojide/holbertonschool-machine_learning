#!/usr/bin/env python3
"""training with momentum"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    creates the training operation for a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm
    :param loss: is the loss of the network
    :param alpha: is the learning rate
    :param beta1: is the momentum weight
    :return: the momentum optimization operation
    """
    return tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)\
        .minimize(loss)

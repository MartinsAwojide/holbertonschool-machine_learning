#!/usr/bin/env python3
"""inverse time decay"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates learning rate decay operation in tensorflow with inverse time decay
    :param alpha: is the original learning rate
    :param decay_rate: weight used determine the rate at which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: number of passes of gradient descent that should
    occur before alpha is decayed further
    :return: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
#!/usr/bin/env python3
"""learning rate decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates the learning rate using inverse time decay in numpy
    :param alpha: is the original learning rate
    :param decay_rate: weight to determine the rate at which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: is the number of passes of gradient descent that
    should occur before alpha is decayed further
    :return: the updated value for alpha
    """
    epoch_number = int(global_step / decay_step)
    alpha = alpha / (1 + (decay_rate * epoch_number))
    return alpha

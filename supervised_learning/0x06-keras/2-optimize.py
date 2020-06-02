#!/usr/bin/env python3
"""build model in keras"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics
    :param network: is the model to optimize
    :param alpha: is the learning rate
    :param beta1: is the first Adam optimization parameter
    :param beta2: is the second Adam optimization parameter
    :return: None
    """
    opt = K.optimizers.Adam(alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=opt,
                    metrics=['accuracy'])
    return None

#!/usr/bin/env python3
"""build model in keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    :param nx: is the number of input features to the network
    :param layers: list containing the number of nodes in each layer of network
    :param activations: list containing the activation functions used for each
    layer of the network
    :param lambtha: is the L2 regularization parameter
    :param keep_prob: is the probability that a node will be kept for dropout
    :return: the keras model
    """
    model = K.Sequential()
    model.add(K.layers.Dense(layers[0], activation=activations[0],
                             kernel_regularizer=K.regularizers.l2(lambtha),
                             input_shape=(nx,)))
    rate = 1 - keep_prob
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(rate))
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lambtha))
                  )
    return model

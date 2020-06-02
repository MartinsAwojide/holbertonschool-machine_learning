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
    K.Model()
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lambtha))(inputs)
    #  Pass inputs to the function x   =>    x = x(inputs)
    rate = 1 - keep_prob
    for i in range(1, len(layers)):
        if i == 1:
            outputs = K.layers.Dropout(rate)(x)
        else:
            outputs = K.layers.Dropout(rate)(outputs)
        outputs = K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lambtha)
                                 )(outputs)
    model = K.Model(inputs=inputs, outputs=outputs)
    return model

#!/usr/bin/env python3
"""Forward Propagation"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network
    :param x: is the placeholder for the input data
    :param layer_sizes: is a list containing the number of nodes in each layer
    of the network
    :param activations: is a list containing the activation functions for each
    layer of the network
    :return: the prediction of the network in tensor form
    """
    for i in range(len(activations)):
        if i == 0:
            yhat = create_layer(x, layer_sizes[i], activations[i])
        else:
            yhat = create_layer(yhat, layer_sizes[i], activations[i])

    return yhat

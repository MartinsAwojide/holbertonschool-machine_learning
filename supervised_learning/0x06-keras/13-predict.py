#!/usr/bin/env python3
"""prediction"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network
    :param network: is the network model to test
    :param data: is the input data to test the model with
    :param verbose: is a boolean that determines if output should be
    printed during the testing process
    :return: the prediction for the data
    """
    return network.predict(x=data, verbose=verbose)

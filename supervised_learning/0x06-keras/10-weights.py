#!/usr/bin/env python3
"""saves & load model"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves an entire model
    :param network: is the model to save
    :param filename: is the path of the file that the model should be saved to
    :param save_format: is the format in which the weights should be saved
    :return: None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    loads an entire model
    :param filename: path of the file that the model should be loaded from
    :return: None
    """
    network.load_weights(filename)
    return None

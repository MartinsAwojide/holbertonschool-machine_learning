#!/usr/bin/env python3
"""saves & load model in json format"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves an entire model
    :param network: is the model to save
    :param filename: is the path of the file that the model should be saved to
    :return: None
    """
    with open(filename, "w") as fd:
        fd.write(network.to_json())
    return None


def load_config(filename):
    """
    loads an entire model
    :param filename: path of the file that the model should be loaded from
    :return: the loaded model
    """
    with open(filename, "r") as fd:
        load = fd.read()
    return K.models.model_from_json(load)

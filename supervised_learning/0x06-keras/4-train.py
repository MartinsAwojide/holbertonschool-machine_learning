#!/usr/bin/env python3
"""train model"""


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """
    Trains a model using mini-batch gradient descent
    :param network: is the model to train
    :param data: ndarray of shape (m, nx) containing the input data
    :param labels: is a one-hot numpy.ndarray of shape (m, classes)
    containing the labels of data
    :param batch_size: size of the batch used for mini-batch gradient descent
    :param epochs:number of passes through data for mini-batch gradient descent
    :param verbose: boolean that determines if output should be printed
    :param shuffle: boolean that determines whether to shuffle the batches
    every epoch. Normally, it is a good idea to shuffle, but for
    reproducibility, we have chosen to set the default to False.
    :return: the History object generated after training the model
    """
    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          shuffle=shuffle, verbose=verbose)
    return history

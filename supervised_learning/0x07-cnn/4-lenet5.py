#!/usr/bin/env python3
"""LeNet-5"""
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow
    Args:
        x: is a tf.placeholder of shape (m, 28, 28, 1) containing the input
        images for the network
            - m is the number of images
        y: is a tf.placeholder of shape (m, 10) containing the one-hot
        labels for the network
    The model consist of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes

    Returns:
            - a tensor for the softmax activated output
            - a training operation that utilizes Adam optimization
            (with default hyperparameters)
            - a tensor for the loss of the netowrk
            - a tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=activation, kernel_initializer=init)(x)

    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=activation,
                             kernel_initializer=init)(pool1)

    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = tf.layers.Flatten()(pool2)

    fc1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    fc2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(fc1)
    fc3 = tf.layers.Dense(units=10, activation=None,
                          kernel_initializer=init)(fc2)

    y_pred = tf.nn.softmax(fc3)
    loss = tf.losses.softmax_cross_entropy(y, fc3)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(fc3, 1))
    accuracy = tf. reduce_mean(tf.cast(correct_prediction, tf.float32))
    train = tf.train.AdamOptimizer().minimize(loss)

    return y_pred, train, loss, accuracy

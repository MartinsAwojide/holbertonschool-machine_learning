#!/usr/bin/env python3
"""
Adam optimization, mini-batch gradient descent, learning rate decay,
and batch normalization neural network model in tensorflow
"""
import tensorflow as tf
import numpy as np


def create_layer(prev, n, activation):
    """
    He et al initialization
    :param prev: is the tensor output of the previous layer
    :param n: is the number of nodes in the layer to create
    :param activation: is the activation function that the layer should use
    :return: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name="layer")
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow
    :param prev: is the activated output of the previous layer
    :param n: is the number of nodes in the layer to be created
    :param activation: is the activation function to be used on the output
    :return: a tensor of the activated output for the layer
    """
    # Layers
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = output(prev)

    # Gamma and Beta initialization
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name="gamma", trainable=True)
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]),
                       name="beta", trainable=True)

    # Batch normalization
    mean, var = tf.nn.moments(Z, axes=0)  # return tensor mean, variance
    b_norm = tf.nn.batch_normalization(Z, mean, var, offset=beta, scale=gamma,
                                       variance_epsilon=1e-8)
    if activation is None:
        return b_norm
    else:
        return activation(b_norm)


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

    for i in range(len(layer_sizes)):
        if i == 0:
            yhat = create_batch_norm_layer(x, layer_sizes[i], activations[i])
        elif i == len(layer_sizes) - 1:
            yhat = create_layer(yhat, layer_sizes[i], activations[i])
        else:
            yhat = create_batch_norm_layer(yhat, layer_sizes[i],
                                           activations[i])

    return yhat


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction
    :param y: is a placeholder for the labels of the input data
    :param y_pred: is a tensor containing the network’s predictions
    :return: a tensor containing the decimal accuracy of the prediction
    """
    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(prediction, tf.float32))


def calculate_loss(y, y_pred):
    """
    a function that calculates the loss of a prediction
    :param y: a placeholders with the right labels of the input data
    :param y_pred: tensor containing the network's predictions
    :return: a tensor containing the loss of a prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way
    :param X:  is the first numpy.ndarray of shape (m, nx) to shuffle
            m is the number of data points
            nx is the number of features in X
    :param Y: is the second numpy.ndarray of shape (m, ny) to shuffle
            m is the same number of data points as in X
            ny is the number of features in Y

    :return: the shuffled X and Y matrices
    """
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    return shuffled_X, shuffled_Y


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization
    :param Data_train: tuple containing the training inputs and training labels
    :param Data_valid: tuple containing the validation inputs and labels
    :param layers: list containing number of nodes in each layer of the network
    :param activations: list containing the activation functions
    used for each layer of the network
    :param alpha: is the learning rate
    :param beta1: weight for the first moment of Adam Optimization
    :param beta2: weight for the second moment of Adam Optimizatio
    :param epsilon: small number used to avoid division by zero
    :param decay_rate: decay rate for inverse time decay of the learning rate
    :param batch_size: number of data points that should be in a mini-batch
    :param epochs: number of times training should pass through whole dataset
    :param save_path: is the path where the model should be saved to
    :return: the path where the model was saved
    """

    # Data(X,Y); X=(50000,784) Y=(50000,10)
    m, nx = Data_train[0].shape
    classes = Data_train[1].shape[1]
    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    # first create a placeholder X, Y
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # we need to print loss and accuracy, for that I need Y_hat

    y_hat = forward_prop(x, layers, activations)
    tf.add_to_collection('y_hat', y_hat)

    accuracy = calculate_accuracy(y, y_hat)
    loss = calculate_loss(y, y_hat)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    # apply learning rate decay to test adam
    global_step = tf.Variable(0, trainable=False)
    change_alpha = tf.train.inverse_time_decay(alpha, global_step, decay_rate,
                                               1, staircase=True)

    train_op = tf.train.AdamOptimizer(change_alpha, beta1,
                                      beta2, epsilon).minimize(loss)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # initialize session and use mini batch gradient descent
    with tf.Session() as sess:
        sess.run(init)
        train = {x: X_train, y: Y_train}
        valid = {x: X_valid, y: Y_valid}
        complete_minibatches = m / batch_size

        if complete_minibatches.is_integer() is True:
            complete_minibatches = int(complete_minibatches)
        else:
            complete_minibatches = (int(complete_minibatches) + 1)

        for i in range(epochs + 1):
            cost_t = sess.run(loss, feed_dict=train)
            acc_t = sess.run(accuracy, feed_dict=train)
            cost_v = sess.run(loss, feed_dict=valid)
            acc_v = sess.run(accuracy, feed_dict=valid)

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))

            if i < epochs:  # number of epochs to do
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)
                for k in range(complete_minibatches):  # mini batches
                    start = k * batch_size  # ex: batch_size=10, start:0,10,20
                    end = (k + 1) * batch_size  # same:10,20,30
                    if end > m:  # end case, goes until m otherwise out range
                        end = m
                    mini_batch_X = shuffled_X[start:end]
                    mini_batch_Y = shuffled_Y[start:end]
                    new_train = {x: mini_batch_X, y: mini_batch_Y}
                    sess.run(train_op, feed_dict=new_train)
                    if (k + 1) % 100 == 0 and k != 0:
                        mb_c, mb_a = sess.run([loss, accuracy], new_train)
                        print("\tStep {}:".format(k + 1))
                        print("\t\tCost: {}".format(mb_c))
                        print("\t\tAccuracy: {}".format(mb_a))
            sess.run(tf.assign(global_step, global_step + 1))
        return saver.save(sess, save_path)

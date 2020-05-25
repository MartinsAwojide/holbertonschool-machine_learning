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

    prediction = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for layer in range(1, len(layer_sizes)):
        if layer != len(layer_sizes) - 1:
            prediction = create_batch_norm_layer(prediction, layer_sizes[
                layer], activations[layer])
        else:
            prediction = create_layer(prediction, layer_sizes[layer],
                                      activations[layer])
    return prediction


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


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates learning rate decay operation in tensorflow with inverse time decay
    :param alpha: is the original learning rate
    :param decay_rate: weight used determine the rate at which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: number of passes of gradient descent that should
    occur before alpha is decayed further
    :return: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    :param loss: is the loss of the network
    :param alpha: is the learning rate
    :param beta1: is the weight used for the first moment
    :param beta2: is the weight used for the second moment
    :param epsilon: is a small number to avoid division by zero
    :return: the Adam optimization operation
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def create_placeholders(nx, classes):
    """
    return two placeholders
    :param nx: the number of feature columns in our data
    :param classes: the number of classes in our classifier
    :return: placeholders named x and y, respectively
    """
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")
    return x, y


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
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x, y = create_placeholders(nx, classes)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)

    global_step = tf.Variable(0)
    alpha_d = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha_d, beta1, beta2, epsilon)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        m = X_train.shape[0]
        # mini batch definition
        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = m // batch_size + 1

        # training loop
        for i in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))

            if i < epochs:
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

                # mini batches
                for b in range(n_batches):
                    start = b * batch_size
                    end = (b + 1) * batch_size
                    if end > m:
                        end = m
                    X_mini_batch = shuffled_X[start:end]
                    Y_mini_batch = shuffled_Y[start:end]

                    next_train = {x: X_mini_batch, y: Y_mini_batch}
                    sess.run(train_op, feed_dict=next_train)

                    if (b + 1) % 100 == 0 and b != 0:
                        loss_mini_batch = sess.run(loss, feed_dict=next_train)
                        acc_mini_batch = sess.run(accuracy,
                                                  feed_dict=next_train)
                        print("\tStep {}:".format(b + 1))
                        print("\t\tCost: {}".format(loss_mini_batch))
                        print("\t\tAccuracy: {}".format(acc_mini_batch))

            # Update of global step variable for each iteration
            sess.run(tf.assign(global_step, global_step + 1))

        return saver.save(sess, save_path)

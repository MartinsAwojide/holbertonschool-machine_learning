#!/usr/bin/env python3
"""mini batch"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model using mini-batch gradient descent
    :param X_train: ndarray of shape (m, 784). m:data points. 784:features
    :param Y_train: one-hot ndarray of shape (m, 10) containing training labels
                    10 is the number of classes the model should classify
    :param X_valid: a ndarray of shape (m, 784) containing the validation data
    :param Y_valid: one-hot ndarray, shape (m, 10) containing validation labels
    :param batch_size: is the number of data points in a batch
    :param epochs: is the number of times the training should pass through
    the whole dataset
    :param load_path: is the path from which to load the model
    :param save_path: the path where the model should be saved after training
    :return: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        train_op = tf.get_collection("train_op")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        train = {x: X_train, y: Y_train}
        valid = {x: X_valid, y: Y_valid}
        m = X_train.shape[0]

        complete_minibatches = m / batch_size  # truncates to floor number
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
        return saver.save(sess, save_path)

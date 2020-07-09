#!/usr/bin/env python3
"""contains the TrainModel class"""
from triplet_loss import TripletLoss
import tensorflow as tf
import tensorflow.keras as K
import numpy as np


class TrainModel:
    """
    TrainModel class
    """
    def __init__(self, model_path, alpha):
        """
        Class constructor
        Args:
            model_path: path to the base face verification embedding model
            alpha:  is the alpha to use for the triplet loss calculation
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)

        A = tf.keras.Input(shape=(96, 96, 3))
        P = tf.keras.Input(shape=(96, 96, 3))
        N = tf.keras.Input(shape=(96, 96, 3))

        output_A = self.base_model(A)
        output_P = self.base_model(P)
        output_N = self.base_model(N)

        outputs = [output_A, output_P, output_N]
        tl = TripletLoss(alpha)
        output = tl(outputs)

        model = K.models.Model([A, P, N], output)
        model.compile(optimizer="adam")
        self.training_model = model

    def train(self, triplets, epochs=5, batch_size=32, validation_split=0.3,
              verbose=True):
        """
        Trains self.training_model
        Args:
            triplets: is a list of numpy.ndarrayscontaining the inputs
            to self.training_model
            epochs: is the number of epochs to train for
            batch_size: is the batch size for training
            validation_split: is the validation split for training
            verbose: is a boolean that sets the verbosity mode

        Returns:  the History output from the training
        """
        history = self.training_model.fit(triplets, batch_size=batch_size,
                                          epochs=epochs, verbose=verbose,
                                          validation_split=validation_split)
        return history

    def save(self, save_path):
        """
        Saves the base embedding model
        Args:
            save_path:  is the path to save the model

        Returns:  the saved model
        """
        K.models.save_model(self.base_model, save_path)
        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calculates the F1 score of predictions
        Args:
            y_true: numpy.ndarray of shape (m,) containing the correct labels
            y_pred: numpy.ndarray of shape (m,) containing the predicted labels

        Returns: f1 score
        """
        TP = np.count_nonzero(y_pred * y_true)
        FP = np.count_nonzero(y_pred * (y_true - 1))
        FN = np.count_nonzero((y_pred - 1) * y_true)
        if TP + FP == 0:
            return 0
        else:
            precision = TP / (TP + FP)
        if TP + FN == 0:
            return 0
        else:
            recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calculates the accuracy of predictions
        Args:
            y_true: numpy.ndarray of shape (m,) containing the correct labels
            y_pred: numpy.ndarray of shape (m,) containing the predicted labels

        Returns: the accuracy
        """
        TP = np.count_nonzero(y_pred * y_true)
        TN = np.count_nonzero((y_pred - 1) * (y_true - 1))
        FP = np.count_nonzero(y_pred * (y_true - 1))
        FN = np.count_nonzero((y_pred - 1) * y_true)
        acc = (TP + TN) / (TP + FN + TN + FP)
        return acc

    def best_tau(self, images, identities, thresholds):
        """
        Calculates the best tau to use for a maximal F1 score
        Args:
            images: numpy.ndarray of shape (m, n, n, 3) containing
             the aligned images for test
            - m is the number of images
            - n is the size of the images
            identities: list containing the identities of each image in images
            thresholds: 1D numpy.ndarray of distance thresholds (tau) to test

        Returns: (tau, f1, acc)
        - tau- the optimal threshold to maximize F1 score
        - f1 - the maximal F1 score
        - acc - the accuracy associated with the maximal F1 score
        """
        distances = []  # squared L2 distance between pairs
        identical = []  # 1 if same identity, 0 otherwise

        def distance(emb1, emb2):
            """Calculate distance"""
            return np.sum(np.square(emb1 - emb2))

        embedded = np.zeros((images.shape[0], 128))

        for i, m in enumerate(images):
            embedded[i] = self.base_model.predict(np.expand_dims(m, axis=0))[0]

        num = len(identities)

        for i in range(num - 1):
            for j in range(i + 1, num):
                distances.append(distance(embedded[i], embedded[j]))
                identical.append(
                    1 if identities[i] == identities[j] else 0)

        distances = np.array(distances)
        identical = np.array(identical)

        f1_scores = [self.f1_score(identical,
                                   distances < t) for t in thresholds]
        acc_scores = [self.accuracy(identical, distances < t) for t in
                      thresholds]

        opt_idx = np.argmax(f1_scores)

        opt_tau = thresholds[opt_idx]

        opt_f1 = f1_scores[opt_idx]

        opt_acc = acc_scores[opt_idx]
        return opt_tau, opt_f1, opt_acc

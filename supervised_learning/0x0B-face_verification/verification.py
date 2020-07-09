#!/usr/bin/env python3
"""contains the FaceVerification class"""
import tensorflow.keras as K
import tensorflow as tf
import numpy as np


class FaceVerification:
    """Class FaceVerification"""
    def __init__(self, model_path, database, identities):
        """
        Class constructor
        Args:
            model_path: is the path to where the face verification
            embedding model is stored
            database: is a numpy.ndarray of shape (d, e) containing all
            the face embeddings in the database
            identities: is a list of length d containing the identities
            corresponding to the embeddings in database
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = K.models.load_model(model_path)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        """
        Calculates the face embedding of images
        Args:
            images: numpy.ndarray of shape (m, n, n, 3) containing
             the aligned images for test
            - m is the number of images
            - n is the size of the images

        Returns: a numpy.ndarray of shape (i, e) containing the
        embeddings where e is the dimensionality of the embeddings
        """
        embedded = np.zeros((images.shape[0], 128))
        for i, m in enumerate(images):
            embedded[i] = self.model.predict(np.expand_dims(m, axis=0))[0]
        return embedded

    def verify(self, image, tau=0.5):
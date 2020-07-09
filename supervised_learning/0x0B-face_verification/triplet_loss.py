#!/usr/bin/env python3
"""Triple loss"""
from tensorflow.keras.layers import Layer
import tensorflow as tf


class TripletLoss(Layer):
    """Class Triple loss"""
    def __init__(self, alpha, **kwargs):
        """
        Class constructor. Sets the public instance attribute alpha
        Args:
            alpha:  is the alpha value used to calculate the triplet loss
        """
        super(TripletLoss, self).__init__(*kwargs)
        self.alpha = alpha

    def triplet_loss(self, inputs):
        """
        Triple loss
        Args:
            inputs: is a list containing the anchor, positive and negative
            output tensors from the last layer of the model, respectively
        Returns:  a tensor containing the triplet loss values
        """
        anchor, positive, negative = inputs
        # Step 1: Compute the (encoding) distance between the anchor and the
        # positive
        pos_dist = tf.reduce_sum((anchor - positive) ** 2, axis=-1)
        # Step 2: Compute the (encoding) distance between the anchor and the
        # negative
        neg_dist = tf.reduce_sum((anchor - negative) ** 2, axis=-1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = pos_dist - neg_dist + self.alpha
        # Step 4: Take the maximum of basic_loss and 0.0. Don't Sum over the
        # training examples.
        loss = tf.maximum(basic_loss, 0)

        return loss

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

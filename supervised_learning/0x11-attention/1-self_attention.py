#!/usr/bin/env python3
"""RNNEncoder class"""
# source: https://towardsdatascience.com/implementing-
# neural-machine-translation-with-attention-using-tensorflow-fc9c6f26155f
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """RNNEncoder class"""
    def __init__(self, units):
        """Class constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """

        Args:
            s_prev: is a tensor of shape (batch, units) containing the
            previous decoder hidden state
            hidden_states:  is a tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder

        Returns: context, weights
        context is a tensor of shape (batch, units) that contains the
        context vector for the decoder
        weights is a tensor of shape (batch, input_seq_len, 1) that contains
        the attention weights

        """
        values = tf.expand_dims(s_prev, 1)

        score = self.V(tf.nn.tanh(self.W(values) + self.U(hidden_states)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

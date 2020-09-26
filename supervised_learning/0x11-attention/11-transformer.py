#!/usr/bin/env python3
"""
Creates encoder for transformer
Source: https://www.tensorflow.org/tutorials/
text/transformer#create_the_transformer
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """
    Class to create an encoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor
        dm: an integer representing the dimensionality of the model
        h: an integer representing the number of heads
        hidden: the number of hidden units in the fully connected layer
        rop_rate: the dropout rate
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Function to create the decoder block for the transformer
        x: a tensor of shape (batch, target_seq_len, dm)containing the
        input to the decoder block
        encoder_output: a tensor of shape (batch, input_seq_len, dm)
        containing the output of the encoder
        training: a boolean to determine if the model is training
        look_ahead_mask: the mask to be applied to the first multi
        head attention layer
        padding_mask: the mask to be applied to the second multi head
        attention layer
        return: a tensor of shape (batch, target_seq_len, dm) containing
        the block’s output
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        output = self.linear(dec_output)

        return output

#!/usr/bin/env python3
"""Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder
    Args:
        input_dims: is an integer containing the dimensions of the model input
        hidden_layers:  is a list containing the number of nodes for each
            hidden layer in the encoder, respectively
            - the hidden layers should be reversed for the decoder
        latent_dims: is an integer containing the dimensions of the latent
        space representation

    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model
    """
    # first encoder to the latent layer
    input_encoder = keras.Input(shape=(input_dims,))
    output = keras.layers.Dense(hidden_layers[0],
                                activation='relu')(input_encoder)
    for i in range(1, len(hidden_layers)):
        output = keras.layers.Dense(hidden_layers[i],
                                    activation='relu')(output)

    latent = keras.layers.Dense(latent_dims,
                                activation='relu')(output)

    encoder = keras.models.Model(inputs=input_encoder, outputs=latent)
    # encoder.summary()

    # then the encoder goes backwards in dimension till input_dim
    input_decoder = keras.Input(shape=(latent_dims,))
    output2 = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder)
    for i in range(len(hidden_layers) - 2, -1, -1):
        output2 = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(output2)
    last_layer = keras.layers.Dense(input_dims, activation='sigmoid')(output2)

    decoder = keras.models.Model(inputs=input_decoder, outputs=last_layer)
    # decoder.summary()
    input_auto = keras.Input(shape=(input_dims,))
    encoder_out = encoder(input_auto)
    decoder_out = decoder(encoder_out)
    auto = keras.models.Model(inputs=input_auto, outputs=decoder_out)
    # auto.summary()
    auto.compile(loss='binary_crossentropy', optimizer='Adam')

    return encoder, decoder, auto

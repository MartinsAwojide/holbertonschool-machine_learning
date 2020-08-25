#!/usr/bin/env python3
"""Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates an autoencoder
    Args:
        input_dims: is a tuple of integers containing the dimensions
        of the model input
        filters: is a list containing the number of filters for each
        convolutional layer in the encoder, respectively
            - the filters should be reversed for the decoder
        latent_dims: is a tuple of integers containing the dimensions
         of the latent space representation

    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model
    """
    # first encoder to the latent layer
    input_encoder = keras.Input(shape=input_dims)
    output = keras.layers.Conv2D(filters=filters[0],
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu')(input_encoder)
    output = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(output)
    for i in range(1, len(filters)):
        output = keras.layers.Conv2D(filters=filters[i],
                                     kernel_size=(3, 3),
                                     padding='same',
                                     activation='relu')(output)
        output = keras.layers.MaxPool2D(pool_size=(2, 2),
                                        padding='same')(output)

    latent = output

    encoder = keras.models.Model(inputs=input_encoder, outputs=latent)
    # encoder.summary()

    # then the encoder goes backwards in dimension till input_dim
    input_decoder = keras.Input(shape=latent_dims)
    output2 = keras.layers.Conv2D(filters=filters[-1],
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation='relu')(input_decoder)
    output2 = keras.layers.UpSampling2D((2, 2))(output2)
    # The second to last convolution should instead use valid padding
    for i in range(len(filters) - 2, 0, -1):
        output2 = keras.layers.Conv2D(filters=filters[i],
                                      kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu')(output2)
        output2 = keras.layers.UpSampling2D((2, 2))(output2)
    # Takes the first on filters = input of last layer
    output2 = keras.layers.Conv2D(filters=filters[0],
                                  kernel_size=(3, 3),
                                  padding='valid',
                                  activation='relu')(output2)
    output2 = keras.layers.UpSampling2D((2, 2))(output2)
    last_layer = keras.layers.Conv2D(filters=input_dims[-1],
                                     kernel_size=(3, 3),
                                     padding='same',
                                     activation='sigmoid')(output2)

    decoder = keras.models.Model(inputs=input_decoder, outputs=last_layer)
    # decoder.summary()
    input_auto = keras.Input(shape=input_dims)
    encoder_out = encoder(input_auto)
    decoder_out = decoder(encoder_out)
    auto = keras.models.Model(inputs=input_auto, outputs=decoder_out)
    # auto.summary()
    auto.compile(loss='binary_crossentropy', optimizer='Adam')

    return encoder, decoder, auto

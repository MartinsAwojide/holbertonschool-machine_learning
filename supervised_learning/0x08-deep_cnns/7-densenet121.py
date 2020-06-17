#!/usr/bin/env python3
"""DenseNet121"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture
    Args:
        growth_rate: is the growth rate
        compression: is the compression factor
        input data will have shape (224, 224,3)
        All weights use he normal initialization

    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()
    k = 2*growth_rate

    norm0 = K.layers.BatchNormalization(axis=3)(X)
    act0 = K.layers.Activation('relu')(norm0)
    C1 = K.layers.Conv2D(filters=k, kernel_size=7,
                         kernel_initializer=init,
                         padding='same', strides=(2, 2))(act0)
    max_pool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(C1)
    dense1, nb_filters = dense_block(max_pool, k, growth_rate, 6)
    transition1, nb_filters = transition_layer(dense1, nb_filters, compression)

    dense2, nb_filters = dense_block(transition1, nb_filters, growth_rate, 12)
    transition2, nb_filters = transition_layer(dense2, nb_filters, compression)

    dense3, nb_filters = dense_block(transition2, nb_filters, growth_rate, 24)
    transition3, nb_filters = transition_layer(dense3, nb_filters, compression)

    dense4, nb_filters = dense_block(transition3, nb_filters, growth_rate, 16)
    avg_pool = K.layers.AveragePooling2D(pool_size=7, padding='same')(dense4)

    FC = K.layers.Dense(activation='softmax',
                        units=1000)(avg_pool)
    model = K.Model(inputs=X, outputs=FC)
    return model

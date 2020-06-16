#!/usr/bin/env python3
"""GoogleNet"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds a inception Network GoogleNet
    Returns: the keras model
    """

    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, kernel_initializer=init,
                            padding='same', strides=(2, 2),
                            activation='relu')(X)
    max_pool = K.layers.MaxPool2D(pool_size=3, strides=(2, 2),
                                  padding='same')(conv1)
    conv2 = K.layers.Conv2D(filters=64, kernel_size=1,
                            kernel_initializer=init, padding='valid',
                            activation='relu')(max_pool)
    conv3 = K.layers.Conv2D(filters=192, kernel_size=3,
                            kernel_initializer=init, padding='same',
                            activation='relu')(conv2)
    max_pool2 = K.layers.MaxPool2D(pool_size=3, strides=(2, 2),
                                   padding='same')(conv3)
    block1 = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])
    block2 = inception_block(block1, [128, 128, 192, 32, 96, 64])
    max_pool3 = K.layers.MaxPool2D(pool_size=3, strides=(2, 2),
                                   padding='same')(block2)
    block3 = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])
    block4 = inception_block(block3, [160, 112, 224, 24, 64, 64])
    block5 = inception_block(block4, [128, 128, 256, 24, 64, 64])
    block6 = inception_block(block5, [112, 144, 288, 32, 64, 64])
    block7 = inception_block(block6, [256, 160, 320, 32, 128, 128])
    max_pool4 = K.layers.MaxPool2D(pool_size=3, strides=(2, 2),
                                   padding='same')(block7)
    block8 = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])
    block9 = inception_block(block8, [384, 192, 384, 48, 128, 128])
    avg_pool = K.layers.AveragePooling2D(pool_size=7, padding='valid',
                                         strides=(1, 1))(block9)
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)
    FC = K.layers.Dense(activation='softmax', kernel_initializer=init,
                        units=1000)(dropout)
    model = K.Model(inputs=X, outputs=FC)
    return model

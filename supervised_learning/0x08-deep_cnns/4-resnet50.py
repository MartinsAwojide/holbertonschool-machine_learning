#!/usr/bin/env python3
"""ResNet-50 architecture"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture
    Returns: the keras model
    """

    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    C1 = K.layers.Conv2D(filters=64, kernel_size=7, kernel_initializer=init,
                         padding='same', strides=(2, 2))(X)
    C1 = K.layers.BatchNormalization(axis=3)(C1)
    C1 = K.layers.Activation('relu')(C1)
    max_pool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(C1)

    Pro1 = projection_block(max_pool, [64, 64, 256], 1)
    Iden1 = identity_block(Pro1, [64, 64, 256])
    Iden1 = identity_block(Iden1, [64, 64, 256])

    Pro2 = projection_block(Iden1, [128, 128, 512])
    Iden2 = identity_block(Pro2, [128, 128, 512])
    Iden2 = identity_block(Iden2, [128, 128, 512])
    Iden2 = identity_block(Iden2, [128, 128, 512])

    Pro3 = projection_block(Iden2, [256, 256, 1024])
    Iden3 = identity_block(Pro3, [256, 256, 1024])
    Iden3 = identity_block(Iden3, [256, 256, 1024])
    Iden3 = identity_block(Iden3, [256, 256, 1024])
    Iden3 = identity_block(Iden3, [256, 256, 1024])
    Iden3 = identity_block(Iden3, [256, 256, 1024])

    Pro4 = projection_block(Iden3, [512, 512, 2048])
    Iden4 = identity_block(Pro4, [512, 512, 2048])
    Iden4 = identity_block(Iden4, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=7, padding='same')(Iden4)

    FC = K.layers.Dense(activation='softmax',
                        units=1000)(avg_pool)
    model = K.Model(inputs=X, outputs=FC)
    return model

#!/usr/bin/env python3
"""
TRansfer learning
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Data preprocessing
    Args:
        X:
        Y:

    Returns:

    """
    X = K.applications.resnet50.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == "__main__":
    input_tensor = K.Input(shape=(32, 32, 3))
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    model_base = K.applications.ResNet101(
                                            include_top=False,
                                            weights="imagenet",
                                            input_tensor=input_tensor)
    model = K.models.Sequential()
    model.add(K.layers.UpSampling2D((8, 8)))
    model.add(model_base)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=K.optimizers.Adam(lr=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=32,
                        epochs=5,
                        verbose=1)
    model.save('cifar10.h5')

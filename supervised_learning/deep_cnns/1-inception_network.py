#!/usr/bin/env python3
"""
Exercice 1 : Inception Network
"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """model consisting of multiple inception
    blocks. It is designed for image classification.
    func is to inputs data of shape given below and
    is to return inception Neural Network model. """
    # Input shape
    inputs = K.Input(shape=(224, 224, 3))

    # inputs layer
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same', activation='relu')(inputs)
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    # conv layer num1
    x = K.layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    # conv layer num2
    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    # inception blocks designed for image classification
    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    # inception 5a and 5b
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    # final Layers
    x = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(7, 7),
                                  padding='valid')(x)
    x = K.layers.Dropout(0.4)(x)  # dropout
    x = K.layers.Dense(1000, activation='softmax')(x)

    # create the model
    model = K.models.Model(inputs=inputs, outputs=x)

    return model

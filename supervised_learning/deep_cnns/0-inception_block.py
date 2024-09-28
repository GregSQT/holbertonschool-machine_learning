#!/usr/bin/env python3
"""
Inception Block
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    function that builds an inception block
    as described in Going Deeper with Convolutions (2014)
    """
    # Initialize the weights using the He normal initialization method
    initializer = K.initializers.he_normal()

    # First branch: 1x1 convolution with filters[0] filters & ReLU activation
    F1_layer = K.layers.Conv2D(filters=filters[0],
                               kernel_size=1,
                               padding='same',
                               kernel_initializer=initializer,
                               activation='relu')
    F1_output = F1_layer(A_prev)

    # Second branch: 1x1 convolution with filters[1] filters & ReLU activation
    F3R_layer = K.layers.Conv2D(filters=filters[1],
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                activation='relu')
    F3R_output = F3R_layer(A_prev)

    # Third branch: 3x3 convolution with filters[2] filters & ReLU activation
    F3_layer = K.layers.Conv2D(filters=filters[2],
                               kernel_size=3,
                               padding='same',
                               kernel_initializer=initializer,
                               activation='relu')
    F3_output = F3_layer(F3R_output)

    # Fourth branch: 1x1 convolution with filters[3] filters & ReLU activation
    F5R_layer = K.layers.Conv2D(filters=filters[3],
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                activation='relu')
    F5R_output = F5R_layer(A_prev)

    # Fifth branch: 5x5 convolution with filters[4] filters & ReLU activation
    F5_layer = K.layers.Conv2D(filters=filters[4],
                               kernel_size=5,
                               padding='same',
                               kernel_initializer=initializer,
                               activation='relu')
    F5_output = F5_layer(F5R_output)

    # Sixth branch: Max pooling with a pool size of 3x3 and a stride of 1
    Pool_layer = K.layers.MaxPool2D(pool_size=3,
                                    padding='same',
                                    strides=1)
    Pool_output = Pool_layer(A_prev)

    # Seventh branch: 1x1 convolution with filters[5]
    # filters and ReLU activation
    FPP_layer = K.layers.Conv2D(filters=filters[5],
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                activation='relu')
    FPP_output = FPP_layer(Pool_output)

    # Concatenate the outputs from all the branches to form the final output
    output = K.layers.concatenate(
        [F1_output, F3_output, F5_output, FPP_output])

    return output

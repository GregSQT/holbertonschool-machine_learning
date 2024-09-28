#!/usr/bin/env python3
"""
Exercice 2 : Identity Block
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    function that builds an identity block
    as described in Deep Residual Learning for Image Recognition (2015)
    """
    # Use He normal initializer for the weights
    initializer = K.initializers.he_normal()

    # First component of the identity block: 1x1 Convolution,
    # BatchNormalization, ReLU Activation
    F11_layer = K.layers.Conv2D(
        filters=filters[0],
        kernel_size=1,
        padding='same',
        kernel_initializer=initializer,
        activation=None)
    F11_output = F11_layer(A_prev)
    F11_norm = K.layers.BatchNormalization()
    F11_output = F11_norm(F11_output)
    F11_activ = K.layers.Activation('relu')
    F11_output = F11_activ(F11_output)

    # Second component of the identity block: 3x3 Convolution,
    # BatchNormalization, ReLU Activation
    F3_layer = K.layers.Conv2D(
        filters=filters[1],
        kernel_size=3,
        padding='same',
        kernel_initializer=initializer,
        activation=None)
    F3_output = F3_layer(F11_output)
    F3_norm = K.layers.BatchNormalization()
    F3_output = F3_norm(F3_output)
    F3_activ = K.layers.Activation('relu')
    F3_output = F3_activ(F3_output)

    # Third component of the identity block: 1x1 Convolution,
    # BatchNormalization
    F12_layer = K.layers.Conv2D(
        filters=filters[2],
        kernel_size=1,
        padding='same',
        kernel_initializer=initializer,
        activation=None)
    F12_output = F12_layer(F3_output)
    F12_norm = K.layers.BatchNormalization()
    F12_output = F12_norm(F12_output)

    # Add the input (residual connection) and output together
    output = K.layers.Add()([F12_output, A_prev])

    # Activate the combined output with ReLU
    output = K.layers.Activation('relu')(output)

    return output

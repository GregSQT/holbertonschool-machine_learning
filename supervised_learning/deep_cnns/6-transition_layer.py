#!/usr/bin/env python3
"""
Dense Block
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    function that builds a transition layer
    as described in Densely Connected Convolutional Networks
    """

    # Use the He normal initializer for the weights
    initializer = K.initializers.he_normal()

    # Apply Batch Normalization to the input X
    l1_norm = K.layers.BatchNormalization()
    l1_output = l1_norm(X)

    # Apply ReLU activation to the normalized output
    l1_activ = K.layers.Activation('relu')
    l1_output = l1_activ(l1_output)

    # Apply a 1x1 Convolution to reduce
    # the number of filters with compression factor
    l1_layer = K.layers.Conv2D(filters=int(nb_filters * compression),
                               kernel_size=1,
                               padding='same',
                               kernel_initializer=initializer,
                               activation=None)
    l1_output = l1_layer(l1_output)

    # Apply Average Pooling with pool_size=2 to reduce the spatial dimensions
    avg_pool = K.layers.AvgPool2D(pool_size=2,
                                  padding='same',
                                  strides=None)
    X = avg_pool(l1_output)

    # Return the output X and the number of filters in the output
    return X, X.shape[-1]
    # Alternatively, you can return the number of filters using:
    # return X, int(nb_filters*compression)
    
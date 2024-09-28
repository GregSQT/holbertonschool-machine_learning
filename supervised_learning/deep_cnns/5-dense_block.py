#!/usr/bin/env python3
"""
Dense Block
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Function that builds a dense block as described
    in Densely Connected Convolutional Networks
    """

    # Use the He normal initializer for the weights
    initializer = K.initializers.he_normal()

    # Loop over the specified number of layers
    for i in range(layers):

        # Batch normalization and ReLU activation for X (input)
        l1_norm = K.layers.BatchNormalization()
        l1_output = l1_norm(X)
        l1_activ = K.layers.Activation('relu')
        l1_output = l1_activ(l1_output)

        # 1x1 Convolution with 4 times the growth rate
        # filters and no activation function applied yet
        l1_layer = K.layers.Conv2D(filters=4 * growth_rate,
                                   kernel_size=1,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   activation=None)
        l1_output = l1_layer(l1_output)

        # Batch normalization and ReLU activation for
        # the output of the first convolution
        l2_norm = K.layers.BatchNormalization()
        l2_output = l2_norm(l1_output)
        l2_activ = K.layers.Activation('relu')
        l2_output = l2_activ(l2_output)

        # 3x3 Convolution with the growth rate number of filters
        # and no activation function applied yet
        l2_layer = K.layers.Conv2D(filters=growth_rate,
                                   kernel_size=3,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   activation=None)
        l2_output = l2_layer(l2_output)

        # Concatenate the outputs of the branches (input & output) channel-wise
        X = K.layers.concatenate([X, l2_output])

        # (Optionally) infer the number of channels
        # in the output (after concatenation)
        # nb_filters += growth_rate

    # (Optionally) activate the combined output with ReLU
    # output = K.layers.Activation('relu')(output)

    # Return the concatenated output and the updated number of filters
    return X, X.shape[-1]
    # Alternatively, this return works as well:
    # return X, nb_filters
    
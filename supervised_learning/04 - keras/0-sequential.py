#!/usr/bin/env python3
"""
Exercice 0 : Sequential - use the Sequential class
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    # Create a Sequential model
    network = K.models.Sequential()

    # Iterate over the layers
    for i in range(len(layers)):
        if i == 0:
            # Add the input layer with specified number of nodes,
            # activation function,
            # L2 regularization, and input shape
            network.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)))
        else:
            # Add the hidden layers with specified number of nodes,
            # activation function,
            # and L2 regularization
            network.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)))

        # Add dropout layer after each hidden layer, except the last one
        if i != len(layers) - 1:
            network.add(K.layers.Dropout(1 - keep_prob))

    # Return the constructed network
    return network

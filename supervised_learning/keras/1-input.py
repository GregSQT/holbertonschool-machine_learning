#!/usr/bin/env python3
"""
Exercice 1 : Input - use the Input class
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    # Define the input layer using the Input class with specified input shape
    inputs = K.layers.Input(shape=(nx,))

    # Iterate over the layers
    for i in range(len(layers)):
        if i == 0:
            # Set the initial outputs to be the inputs
            outputs = inputs

        # Add a dense layer with specified number of nodes,
        # activation function,
        # and L2 regularization, and connect it to the previous outputs
        outputs = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(outputs)

        # Add dropout layer after each hidden layer, except the last one
        if i != len(layers) - 1:
            outputs = K.layers.Dropout(1 - keep_prob)(outputs)

    # Create a Model with defined inputs and outputs
    network = K.models.Model(inputs=inputs, outputs=outputs)

    # Return the constructed network
    return network
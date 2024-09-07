#!/usr/bin/env python3
"""
Exercice 2 : Optimize
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """function that sets up an Adam optimizer"""
    # Create an Adam optimizer with specified learning rate, beta1, beta2
    optimizer = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)

    # Define the loss function as categorical cross-entropy
    loss = 'categorical_crossentropy'

    # Compile the network with optimizer, loss function, and accuracy metric
    network.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Return None
    return None

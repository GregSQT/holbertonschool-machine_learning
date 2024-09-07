#!/usr/bin/env python3
"""
Exercicei 5 : Train a model, with validation data
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""
    # Train the network using the fit method of the network object
    # Pass the data and labels, along with other training parameters
    # Include the validation data if provided
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data)

    # Return history object, which contains training metrics and loss values
    return history

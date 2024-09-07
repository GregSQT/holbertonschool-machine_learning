#!/usr/bin/env python3
"""
Exercice 13 : Predict
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Function that makes a prediction using a neural network"""
    result = network.predict(
        x=data,  # Input data for prediction
        batch_size=None,  # Batch size for prediction
        # (default is None for automatic batch size)
        verbose=verbose  # Whether to display
        # prediction progress (default is False)
    )
    return result  # Return the prediction result

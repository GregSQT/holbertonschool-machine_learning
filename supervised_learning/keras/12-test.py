#!/usr/bin/env python3
"""
Exercice 12 : Test
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Function that tests a neural network"""
    result = network.evaluate(
        x=data,  # Input data for evaluation
        y=labels,  # True labels for evaluation
        batch_size=None,  # Batch size for evaluation
        # (default is None for automatic batch size)
        verbose=verbose
        # Whether to display evaluation progress (default is True)
    )
    return result  # Return the evaluation result

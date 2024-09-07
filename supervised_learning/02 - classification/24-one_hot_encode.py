#!/usr/bin/env python3
""" Provides converting a numeric label vector into a one-hot matrix."""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y: numpy.ndarray with shape (m,) containing numeric class labels.
        classes: maximum number of classes found in Y (integer).

    Returns:
        One-hot encoding of Y with shape (classes, m), or None on failure.
    """
    # Check if Y is a valid numpy array with shape (m,)
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None

    # Check if classes is an integer
    if not isinstance(classes, int):
        return None

    # Get the number of examples (m)
    m = Y.shape[0]

    # Create an array of zeros with shape (classes, m)
    one_hot = np.zeros((classes, m))

    # Iterate over the label vector Y
    for i, label in enumerate(Y):
        # Check if the label is within the valid range
        if label < 0 or label >= classes:
            return None
        # Set the corresponding element in one_hot to 1
        one_hot[label, i] = 1

    # Return the resulting one-hot encoding
    return one_hot

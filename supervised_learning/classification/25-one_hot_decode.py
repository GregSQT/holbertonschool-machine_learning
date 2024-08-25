#!/usr/bin/env python3
""" Provides converting a numeric label vector into a one-hot matrix."""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m).
                 classes: maximum number of classes.
                 m: number of examples.

    Returns:
        numpy.ndarray with shape (m,) containing the numeric labels for e.g,
        or None on failure.
    """
    # Check if one_hot is a valid numpy array with shape (classes, m)
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    # Get the number of examples (m)
    m = one_hot.shape[1]

    # Initialize an empty label vector
    labels = np.zeros((m,), dtype=int)

    # Iterate over the examples
    for i in range(m):
        # Find the index of the maximum value in each example (column-wise)
        max_index = np.argmax(one_hot[:, i])

        # Set the corresponding label to the maximum index
        labels[i] = max_index

    # Return the resulting label vector
    return labels

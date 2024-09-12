#!/usr/bin/env python3
"""
Normalization Constants
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants(mean & std deviation) of a matrix.

    Args:
        X (numpy.ndarray): Input matrix to normalize.

    Returns:
        numpy.ndarray: Array containing the mean of each feature.
        numpy.ndarray: Array containing the standard deviation of each feature
    """
    m = X.shape[0]  # Number of data points
    mean = np.sum(X, axis=0) / m  # Calculate the mean along the columns
    # Calculate the standard deviation
    stddev = np.sqrt(np.sum((X - mean) ** 2, axis=0) / m)

    return mean, stddev

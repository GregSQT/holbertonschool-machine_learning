#!/usr/bin/env python3
"""
Exercice 13 : Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """function that normalizes unactivated output by batch normalization"""

    # Compute the mean and standard deviation of Z
    m, stddev = normalization_constants(Z)

    # Compute the variance from the standard deviation
    s = stddev ** 2

    # Normalize Z using the batch normalization formula
    Z_norm = (Z - m) / np.sqrt(s + epsilon)

    # Scale and shift Z_norm using gamma and beta
    Z_b_norm = gamma * Z_norm + beta

    # Return the normalized and scaled Z matrix
    return Z_b_norm


def normalization_constants(X):
    """function that calculates the normalization constants of a matrix"""

    # Get the number of data points (m)
    m = X.shape[0]

    # Calculate the mean along axis 0
    mean = np.sum(X, axis=0) / m

    # Calculate the standard deviation along axis 0
    stddev = np.sqrt(np.sum((X - mean) ** 2, axis=0) / m)

    # Return the mean and standard deviation
    return mean, stddev

#!/usr/bin/env python3
"""
Exercice 1 : Normalize
"""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix based on mean and standard deviation.

    Args:
        X (numpy.ndarray): Input matrix to normalize.
        m (numpy.ndarray): Array containing the mean of each feature.
        s (numpy.ndarray): Array containing the standard deviation each fea..

    Returns:
        numpy.ndarray: The normalized X matrix.
    """
    X_norm = (X - m) / s  # Element-wise subtraction and division
    return X_norm

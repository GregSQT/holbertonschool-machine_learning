#!/usr/bin/env python3
"""
Mean and Covariance (calculator)
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance matrix of a data set.

    Args:
        X (np.ndarray): dataset of shape (n, d)

    Returns:
        Tuple containing the mean and covariance matrix of the dataset.
    """
    # Check if X is a numpy.ndarray and has the correct shape
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    # Check if X contains multiple data points
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape

    # Calculate the mean of X
    mean = np.mean(X, axis=0).reshape((1, d))

    ones = np.ones((n, n))

    # Standardize the data
    std_scores = X - np.matmul(ones, X) * (1 / n)

    # Calculate the covariance matrix
    cov_matrix = np.matmul(std_scores.T, std_scores) / (n - 1)

    return mean, cov_matrix

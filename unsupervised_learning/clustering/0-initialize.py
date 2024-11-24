#!/usr/bin/env python3
"""
0-initialize.py
"""
# Import NumPy library
import numpy as np


# Define a function called 'initialize' that takes two arguments: 'X' and 'k'
def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): The input dataset with shape
        (n_samples, n_features).
        k (int): The number of clusters.

    Returns:
        numpy.ndarray: The initialized centroids for each cluster.
    """

    # Check if 'X' is a NumPy array with two dimensions
    # and return None if not
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    # Get the number of data samples (n_samples)
    # and the number of features (n_features) in the dataset
    n_samples, n_features = X.shape

    # Check if 'k' is a positive integer within a valid range
    # and return None if not
    if not isinstance(k, int) or k <= 0 or k > n_samples:
        return None

    # Initialize centroids as an array of random values within the data range
    centroids = np.random.uniform(np.min(X, axis=0),
                                  np.max(X, axis=0), (k, n_features))

    # Return the initialized centroids
    return centroids

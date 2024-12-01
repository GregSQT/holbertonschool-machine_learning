#!/usr/bin/env python3
"""
variance function
"""
import numpy as np


def variance(X, C):
    """
    Calculate the total intra-cluster variance for a data set.

    Args:
        X (numpy.ndarray): The data set with shape (n_samples, n_features).
        C (numpy.ndarray): The centroid means for each
        cluster with shape (n_clusters, n_features).

    Returns:
        float: The total intra-cluster variance, or None on failure.
    """
    try:
        # Get the dimensions of the input arrays
        num_clusters, num_features = C.shape
        num_samples, _ = X.shape

        # Compute the squared Euclidean distances between data points and
        # centroids
        centroid_squared_norm = np.sum(C ** 2, axis=1)[:, np.newaxis]
        data_squared_norm = np.sum(X ** 2, axis=1)
        dot_product = np.matmul(C, X.T)
        squared_distances = centroid_squared_norm - \
            2 * dot_product + data_squared_norm

        # Calculate the total intra-cluster variance
        total_variance = np.sum(np.amin(squared_distances, axis=0))

        return total_variance

    except Exception:
        return None

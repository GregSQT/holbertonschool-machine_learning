#!/usr/bin/env python3
"""
A function to initialize variables for a Gaussian Mixture Model (GMM)
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): The input dataset with
        shape (n_samples, n_features) containing
        the dataset for K-means clustering.
            - n_samples: The number of data points.
            - n_features: The number of dimensions for each data point.
        k (int): A positive integer containing the number of clusters.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        A tuple containing:
            - priors (numpy.ndarray): The priors for each
            cluster, initialized evenly.
            - centroids (numpy.ndarray): The centroid means for each
            cluster, initialized with K-means.
            - covariances (numpy.ndarray): The covariance matrices for each
            cluster, initialized as identity matrices.
            Returns (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None, None

    n_samples, n_features = X.shape
    priors = np.tile(1 / k, (k,))
    centroids, _ = kmeans(X, k)
    identity_matrix = np.identity(n_features)
    covariances = np.tile(identity_matrix, (k, 1, 1))

    return priors, centroids, covariances

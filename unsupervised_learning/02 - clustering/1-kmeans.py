#!/usr/bin/env python3
"""
1-kmeans
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): The input dataset with
        shape (n_samples, n_features).
        k (int): The number of clusters.
        iterations (int): The maximum number of iterations (default is 1000).

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing:
            - C: A numpy.ndarray of shape (k, n_features)
            containing the centroid means for each cluster.
            - clss: A numpy.ndarray of shape (n_samples,)
            containing the index of the cluster in C
            that each data point belongs to.
              If the function fails, it returns (None, None).
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialization
    n_samples, n_features = X.shape

    # Initialize centroids using the 'initialize' function
    centroids = initialize(X, k)

    if centroids is None:
        return None, None

    prev_centroids = np.copy(centroids)

    X_ = X[:, :, np.newaxis]
    centroids_ = centroids.T[np.newaxis, :, :]
    diff = X_ - centroids_
    distances = np.linalg.norm(diff, axis=1)

    cluster_assignments = np.argmin(distances, axis=1)

    for _ in range(iterations):

        for j in range(k):
            # Recalculate centroids
            cluster_indices = np.where(cluster_assignments == j)
            if len(cluster_indices[0]) == 0:
                centroids[j] = initialize(X, 1)
            else:
                centroids[j] = np.mean(X[cluster_indices], axis=0)

        X_ = X[:, :, np.newaxis]
        centroids_ = centroids.T[np.newaxis, :, :]
        diff = X_ - centroids_
        distances = np.linalg.norm(diff, axis=1)

        cluster_assignments = np.argmin(distances, axis=1)

        if (centroids == prev_centroids).all():
            return centroids, cluster_assignments
        prev_centroids = np.copy(centroids)

    return centroids, cluster_assignments


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

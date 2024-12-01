#!/usr/bin/env python3
"""
3-optimum
"""
import numpy as np


# Import kmeans function from 'kmeans_module'
# and variance function from 'variance_module'
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


# Define the 'optimum_k' function with parameters

def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Function that tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): The input dataset with
        shape (n_samples, n_features).
        kmin (int): The minimum number of clusters to check for (inclusive).
        kmax (int): The maximum number of clusters to check for (inclusive).
        iterations (int): The maximum number of iterations for K-means.

    Returns:
        Tuple[Optional[list], Optional[list]]: A tuple containing:
            - A list of tuples, where each tuple contains cluster centroids
              (centroids) and data point cluster assignments
              (assignments) for a specific cluster size.
            - A list of differences in variance from the smallest cluster size
              for each cluster size. Returns (None, None) on failure.
    """

    # Check if 'X' is a NumPy array with two dimensions
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    # Get the number of data samples (n_samples)
    # and the number of features (n_features)
    n_samples, n_features = X.shape

    # If 'kmax' is not specified, set it to the number of data samples
    if kmax is None:
        kmax = n_samples

    # Check if 'kmin' is a positive integer within a valid range
    if not isinstance(kmin, int) or kmin <= 0 or n_samples <= kmin:
        return None, None

    # Check if 'kmax' is a positive integer within a valid range
    if not isinstance(kmax, int) or kmax <= 0 or n_samples < kmax:
        return None, None

    # Check if 'kmin' is less than 'kmax'
    if kmin >= kmax:
        return None, None

    # Check if 'iterations' is a positive integer
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize lists to store results and delta variances
    results = []
    delta_variances = []

    # Iterate over the number of clusters in the specified range
    for num_clusters in range(kmin, kmax + 1):

        # Perform K-means clustering and store centroids and assignments
        centroids, assignments = kmeans(X, num_clusters, iterations)
        results.append((centroids, assignments))

        # Compute the variance of the clustering result
        variance_value = variance(X, centroids)

        # If it's the first iteration, store the initial variance
        if num_clusters == kmin:
            initial_variance = variance_value

        # Calculate and store the delta variance
        delta_variances.append(np.abs(initial_variance - variance_value))

    # Return the results and delta variances
    return results, delta_variances

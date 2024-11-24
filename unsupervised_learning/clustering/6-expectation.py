#!/usr/bin/env python3
"""
6-expectation - a function def expectation(X, pi, m, S): that calculates
the expectation step in the EM algorithm for a GMM
"""
import numpy as np


# Import the pdf function from the 5-pdf module
pdf = __import__('5-pdf').pdf

def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm
    for a Gaussian Mixture Model (GMM).

    Args:
        X (numpy.ndarray): Data points of shape (n_samples, n_features).
        pi (numpy.ndarray): Priors for each cluster of shape (k,).
        m (numpy.ndarray): Centroid means for each
        cluster of shape (k, n_features).
        S (numpy.ndarray): Covariance matrices for each
        cluster of shape (k, n_features, n_features).

    Returns:
        Tuple[numpy.ndarray, float]:
        - posterior_probs (numpy.ndarray):
        Posterior probabilities for each data point
        in each cluster of shape (k, n_samples).
        - total_log_likelihood (float): Total log likelihood.
        Returns (None, None) on failure.
    """
    # Check if X is a 2D numpy array
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    # Check if pi is a 1D numpy array
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    # Check if m is a 2D numpy array
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    # Check if S is a 3D numpy array
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    num_samples, num_features = X.shape

    # Check if the number of clusters is greater than the number of samples
    if pi.shape[0] > num_samples:
        return None, None

    num_clusters = pi.shape[0]

    # Check if the dimensions of m and S match the number of features and clusters
    if m.shape[0] != num_clusters or m.shape[1] != num_features:
        return None, None

    if S.shape[0] != num_clusters or S.shape[1] != \
    num_features or S.shape[2] != num_features:
        return None, None

    # Check if the sum of priors is close to 1
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    # Initialize an array for posterior probabilities
    posterior_probs = np.zeros((num_clusters, num_samples))

    # Calculate posterior probabilities for each cluster
    for i in range(num_clusters):
        PDF = pdf(X, m[i], S[i])
        posterior_probs[i] = pi[i] * PDF

    # Normalize posterior probabilities
    sum_posterior_probs = np.sum(posterior_probs, axis=0, keepdims=True)
    posterior_probs /= sum_posterior_probs

    # Calculate total log likelihood
    total_log_likelihood = np.sum(np.log(sum_posterior_probs))

    return posterior_probs, total_log_likelihood

#!/usr/bin/env python3
"""
7-maximization - a function def maximization(X, g):
that computes the maximization step in the EM algorithm for a GMM
"""
import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for
    a Gaussian Mixture Model (GMM).

    Args:
        X (numpy.ndarray): Data points of shape (n_samples, n_features).
        g (numpy.ndarray): Posterior probabilities for each data point
        in each cluster of shape (n_clusters, n_samples).

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        - updated_priors (numpy.ndarray): Updated priors for each
        cluster of shape (n_clusters,).
        - updated_means (numpy.ndarray): Updated centroid means for each
        cluster of shape (n_clusters, n_features).
        - updated_covariances (numpy.ndarray): Updated covariance
        matrices for each cluster of
        shape (n_clusters, n_features, n_features).

    Returns (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n_samples, n_features = X.shape
    n_clusters = g.shape[0]

    if g.shape[1] != n_samples:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), np.ones(n_samples)).all():
        return None, None, None

    updated_priors = np.sum(g, axis=1) / n_samples

    updated_means = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    updated_covariances = np.zeros((n_clusters, n_features, n_features))

    for i in range(n_clusters):
        diff = X - updated_means[i]
        updated_covariances[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    return updated_priors, updated_means, updated_covariances

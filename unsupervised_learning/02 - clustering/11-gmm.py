#!/usr/bin/env python3
"""
11-gmm - a function def gmm(X, k): that calculates a GMM from a dataset
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculate a Gaussian Mixture Model (GMM) from a dataset.

    Args:
        X (numpy.ndarray): Input dataset of shape (n_samples, n_features).
        k (int): Number of clusters.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray]:
            - cluster_priors (numpy.ndarray): Cluster priors of shape (k,).
            - cluster_means (numpy.ndarray):
            Centroid means of shape (k, n_features).
            - cluster_covariances (numpy.ndarray):
            Covariance matrices of shape (k, n_features, n_features).
            - cluster_indices (numpy.ndarray):
            Index of the cluster for each data point, shape (n_samples,).
            - bic_scores (numpy.ndarray): Bayesian Information Criterion (BIC)
            for different cluster sizes, shape (kmax - kmin + 1,).
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k)

    gmm_params = gmm.fit(X)
    cluster_indices = gmm.predict(X)
    cluster_priors = gmm_params.weights_
    cluster_means = gmm_params.means_
    cluster_covariances = gmm_params.covariances_
    bic_scores = gmm.bic(X)

    return cluster_priors, cluster_means, cluster_covariances, \
        cluster_indices, bic_scores

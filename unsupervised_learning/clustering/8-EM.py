#!/usr/bin/env python3
"""
A function def expectation_maximization(X, k, iterations=1000,
tol=1e-5, verbose=False) that performs
the expectation maximization for a GMM.
"""
import numpy as np


initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Perform the Expectation-Maximization (EM) algorithm
    for a Gaussian Mixture Model (GMM).

    Args:
        X (numpy.ndarray): Data points of shape (n_samples, n_features).
        k (int): Number of clusters.
        iterations (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance of the log likelihood, used for early stopping.
        verbose (bool): Determines if information about
        the algorithm should be printed.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.
        ndarray, numpy.ndarray, float]:
        - updated_priors (numpy.ndarray):
        Updated priors for each cluster of shape (k,).
        - updated_means (numpy.ndarray): Updated centroid
        means for each cluster of shape (k, n_features).
        - updated_covariances (numpy.ndarray): Updated covariance matrices
        for each cluster of shape (k, n_features, n_features).
        - posterior_probs (numpy.ndarray): Probabilities for each
        data point in each cluster of shape (k, n_samples).
        - log_likelihood (float): Log likelihood of the model.

    Returns (None, None, None, None, None) on failure.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    num_samples, num_features = X.shape
    prev_likelihood = 0
    priors, means, covariances = initialize(X, k)
    responsibilities, likelihood = expectation(X, priors, means, covariances)

    for iteration in range(iterations):
        if verbose and (iteration % 10 == 0):
            print('Log Likelihood after {} iterations: {}'.format(
                iteration, likelihood.round(5)))

        priors, means, covariances = maximization(X, responsibilities)
        responsibilities, likelihood = expectation(
            X, priors, means, covariances)

        if abs(prev_likelihood - likelihood) <= tol:
            break

        prev_likelihood = likelihood

    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            iteration + 1, likelihood.round(5)))

    return priors, means, covariances, responsibilities, likelihood

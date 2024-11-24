#!/usr/bin/env python3
"""
6-pdf - a function def pdf(X, m, S): that calculates the probability
density function of a Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function (PDF) of
    a Gaussian distribution.

    Args:
        X (numpy.ndarray): Data points of shape (n_samples, n_features).
        m (numpy.ndarray): Mean of the distribution of shape (n_features,).
        S (numpy.ndarray): Covariance matrix of shape (n_features, n_features).

    Returns:
        numpy.ndarray: PDF values for each data point of shape (n_samples,).
        Returns None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n_samples, n_features = X.shape

    inverse_covariance = np.linalg.inv(S)
    determinant_covariance = np.linalg.det(S)

    denominator = np.sqrt(((2 * np.pi) ** n_features) * determinant_covariance)

    difference = X.T - m[:, np.newaxis]

    intermediate_matrix = np.matmul(inverse_covariance, difference)
    squared_exponents = np.sum(difference * intermediate_matrix, axis=0)
    exponents = -0.5 * squared_exponents

    pdf_values = np.exp(exponents) / denominator

    pdf_values = np.maximum(pdf_values, 1e-300)

    return pdf_values
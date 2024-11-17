#!/usr/bin/env python3

"""
A module that provides a function for performing Principal
Component Analysis (PCA) on a dataset.
"""
import numpy as np


def pca(X, ndim):
    """
    Perform Principal Component Analysis (PCA) on a dataset.

    Args:
        X (numpy.ndarray): The input dataset with
        shape (n_samples, n_features).
        ndim (int): The number of dimensions for the transformed data.

    Returns:
        numpy.ndarray: The transformed data with reduced dimensionality.
    """
    # Mean Centering

    # Calculate the mean of each feature (column) across all samples
    X_mean = np.mean(X, axis=0)

    # Subtract the mean from the dataset to center it
    X_centered = X - X_mean

    # Singular Value Decomposition (SVD)

    # Perform SVD on the centered dataset
    U, S, VT = np.linalg.svd(X_centered)

    # Reduce Dimensionality

    # Extract the desired number of principal components from VT
    V = VT.T
    U = V[:, :ndim]

    # Project the centered data onto the principal components
    X_pca = np.matmul(X_centered, U)

    # Return the transformed dataset
    return X_pca

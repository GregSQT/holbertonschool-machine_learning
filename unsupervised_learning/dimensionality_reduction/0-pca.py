#!/usr/bin/env python3

"""
PCA
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset using Singular Value Decomposition (SVD).

    Args:
        X (numpy.ndarray): The input dataset.
        var (float): The variance threshold for retaining principal components.

    Returns:
        numpy.ndarray: The principal components matrix.
    """
    # Define a nested function to perform PCA using SVD
    def pca_svd(X, var=0.95):
        # Compute the Singular Value Decomposition (SVD) of the input dataset X
        U, S, Vt = np.linalg.svd(X)

        # Compute the cumulative sum of singular values
        cumulative_variance = np.cumsum(S) / np.sum(S)

        # Determine the number of principal components to retain based on
        # variance threshold
        r = np.min(np.where(cumulative_variance >= var))

        # Get the principal components (Vr)
        V = Vt.T
        Vr = V[..., :r + 1]

        return Vr  # Return the principal components matrix

    # Call the nested function and return the result
    return pca_svd(X, var)

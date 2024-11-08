#!/usr/bin/env python3
"""
Matrix Definiteness
"""

# Import the NumPy library as np
import numpy as np


# Define a function called definiteness that calculates the definiteness
# of a matrix
def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix (np.ndarray): The input matrix.

    Returns:
        str: The definiteness type
        (e.g., 'Positive definite', 'Negative definite', etc.)
             or None if the matrix is invalid.
    """

    # Check if the input matrix is not a NumPy ndarray, and raise a TypeError
    # if so
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if the input matrix is not 2-dimensional, and return None if so
    if matrix.ndim != 2:
        return None

    # Get the dimensions of the matrix
    n, m = matrix.shape

    # Check if the matrix is not square or is empty, and return None if so
    if n != m or n == 0:
        return None

    # Check if the matrix is not symmetric, and return None if so
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate the eigenvalues of the matrix
    eigenvalues = np.linalg.eigvals(matrix)

    # Check various conditions to determine the definiteness type
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"

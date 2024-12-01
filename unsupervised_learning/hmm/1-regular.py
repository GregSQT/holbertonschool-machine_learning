#!/usr/bin/env python3
"""
the function def regular(P): that determines
the steady state probabilities of a regular markov chain
"""
import numpy as np


def regular(P):
    """
    Calculate the steady-state probabilities of a regular Markov chain.

    Args:
        P (numpy.ndarray): Transition matrix of shape (n, n),
            where P[i, j] is the probability of transitioning
            from state i to state j.

    Returns:
        numpy.ndarray: Steady-state probabilities of shape (1, n),
        or None on failure.
    """
    # Check if P is a valid 2D numpy array
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None

    # Check if P is a square matrix
    n, m = P.shape
    if n != m:
        return None

    # Check if the sum of transition probabilities
    # along rows is approximately 1
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, 1):
        return None

    # Calculate the steady-state probabilities using the eigenvalue method
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find the index of the eigenvalue closest to one
    index = np.argmax(np.isclose(eigenvalues, 1))

    # Check if an eigenvalue equal to one is found
    if not np.isclose(eigenvalues[index], 1):
        return None

    # Get the corresponding eigenvector
    steady_state = eigenvectors[:, index]

    # Check if any element in the eigenvector is close to zero
    if np.isclose(steady_state, 0).any():
        return None

    # Normalize the steady-state probabilities
    steady_state /= np.sum(steady_state)

    # Reshape steady_state to have shape (1, n)
    steady_state = steady_state[np.newaxis, :]

    return steady_state

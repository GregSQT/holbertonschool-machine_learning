#!/usr/bin/env python3
"""
the function def markov_chain(P, s, t=1): that determines the
probability of a markov chain being in a particular state after
a specified number of iterations
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Calculate the probability of a Markov chain being
    in a particular state after a specified number of iterations.

    Args:
        P (numpy.ndarray): Transition matrix of shape (n, n),
            where P[i, j] is the probability of transitioning
            from state i to state j.
        s (numpy.ndarray): Initial state probabilities of shape (1, n).
        t (int): Number of iterations.

    Returns:
        numpy.ndarray: Probabilities of being in each state after t iterations
        with shape (1, n), or None on failure.
    """
    # Check if P is a valid 2D numpy array with square shape
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != \
            P.shape[1]:
        return None

    # Check if s is a valid 2D numpy array with shape (1, n)
    if not isinstance(s, np.ndarray) or s.ndim != 2 or s.shape[0] != \
            1 or s.shape[1] != P.shape[0]:
        return None

    # Check if t is a non-negative integer
    if not isinstance(t, int) or t < 0:
        return None

    # Check if sum of transition probabilities along rows is approximately 1
    if not np.allclose(np.sum(P, axis=1), 1):
        return None

    # Check if sum of initial state probabilities is approximately 1
    if not np.allclose(np.sum(s, axis=1), 1):
        return None

    # Initialize the result with the initial state probabilities
    result = s.copy()

    # Perform matrix multiplication t times to simulate t iterations
    for _ in range(t):
        result = np.dot(result, P)

    return result

#!/usr/bin/env python3
"""Intersection"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining this data given various
    hypothetical probabilities of developing severe side effects.

    Args:
        x (int): Number of patients experiencing severe side effects.
        n (int): Total number of patients observed.
        P (np.ndarray): Array of hypothetical probabilities.

    Returns:
        np.ndarray: Array containing the intersection of obtaining
        the data for each probability.

    Raises:
        ValueError: If n is not a positive integer,
        if x is not a valid integer,
        if x > n, or if any value in P or Pr is not in [0, 1],
        or if Pr does not sum to 1.
    """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute the binomial coefficient manually
    binomial_coefficient = (np.math.factorial(
        n) / (np.math.factorial(x) * np.math.factorial(n - x)))

    # Calculate the likelihood for each probability in P
    intersection_values = (
        binomial_coefficient * (P ** x) * ((1 - P) ** (n - x))
    ) * Pr

    return intersection_values

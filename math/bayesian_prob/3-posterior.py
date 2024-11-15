#!/usr/bin/env python3

"""
posterior
a function to calculate the posterior probability for various hypothetical
probabilities of developing severe side effects given the data.

"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculate the posterior probability for various hypothetical
    probabilities of developing severe side effects given observed data.

    Args:
        x (int): Number of patients experiencing severe side effects.
        n (int): Total number of patients observed.
        P (np.ndarray): Array of hypothetical probabilities.
        Pr (np.ndarray): Array containing prior beliefs about P.

    Returns:
        np.ndarray: Array containing the posterior probabilities for each
        probability in P.

    Raises:
        ValueError: If n is not a positive integer,
        if x is not a valid integer,
                    if x > n, or if any value in P or Pr is not in [0, 1],
                    or if Pr does not sum to 1.
        TypeError: If P is not a 1D numpy.ndarray, or Pr is not
                    a numpy.ndarray with the same shape as P.

    """

    # Check if n is a positive integer
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is a non-negative integer
    if not isinstance(x, (int, float)) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    # Check if x is greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if Pr is a numpy.ndarray with the same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Check if all values in P are in the range [0, 1]
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Check if all values in Pr are in the range [0, 1]
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Check if the sum of values in Pr is approximately equal to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Initialize an empty array to store posterior probabilities
    posterior_probabilities = np.zeros_like(P, dtype=float)

    # Compute the binomial coefficient manually
    binomial_coefficient = (np.math.factorial(
        n) / (np.math.factorial(x) * np.math.factorial(n - x)))

    # Calculate the likelihood for each probability in P
    for i, p in enumerate(P):
        posterior_probabilities[i] = (
            binomial_coefficient * (p ** x) * ((1 - p) ** (n - x)) * Pr[i]
        )

    # Normalize posterior probabilities to sum to 1
    posterior_probabilities /= np.sum(posterior_probabilities)

    return posterior_probabilities

#!/usr/bin/env python3
"""
the function def forward(Observation, Emission, Transition, Initial):
that performs the forward algorithm for a hidden markov model
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Perform the forward algorithm for a Hidden Markov Model (HMM).

    Args:
        Observation (numpy.ndarray): 1D array of shape (T,)
        containing the indices of observations.
        Emission (numpy.ndarray): 2D array of shape (N, M)
        with emission probabilities.
        Transition (numpy.ndarray): 2D array of shape (N, N)
        with transition probabilities.
        Initial (numpy.ndarray): 2D array of shape (N, 1)
        containing the initial state probabilities.

    Returns:
        Tuple[float, numpy.ndarray]: Likelihood of the
        observations given the model (P) and
        the forward path probabilities (F).
        Returns (None, None) on failure.
    """
    # Check if Observation is a valid numpy.ndarray with a single dimension
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    # Check if Emission is a valid numpy.ndarray with two dimensions
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    # Check if Transition is a valid numpy.ndarray with two dimensions
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    # Check if Initial is a valid numpy.ndarray with two dimensions
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    # Get the number of observations (T) and the dimensions of model (N, M)
    T = Observation.shape[0]
    N, M = Emission.shape

    # Check if Transition and Initial have the correct shapes
    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    # Check if emission probabilities sum to 1 for each state
    if not np.all(np.isclose(np.sum(Emission, axis=1), np.ones(N))):
        return None, None

    # Check if transition probabilities sum to 1 for each state
    if not np.all(np.isclose(np.sum(Transition, axis=1), np.ones(N))):
        return None, None

    # Check if initial state probabilities sum to 1
    if not np.isclose(np.sum(Initial), 1):
        return None, None

    # Initialize the forward matrix F with zeros
    F = np.zeros((N, T))

    # Calculate the initial probabilities based on the first observation
    Obs_i = Observation[0]
    prob = np.multiply(Initial[:, 0], Emission[:, Obs_i])
    F[:, 0] = prob

    # Perform the forward algorithm for the remaining observations
    for i in range(1, T):
        Obs_i = Observation[i]
        state = np.matmul(F[:, i - 1], Transition)
        prob = np.multiply(state, Emission[:, Obs_i])
        F[:, i] = prob

    # Calculate the likelihood of the observations given the model
    P = np.sum(F[:, T - 1])

    return P, F

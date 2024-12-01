#!/usr/bin/env python3
"""
the function def viterbi(Observation, Emission, Transition, Initial):
that calculates the most likely sequence of hidden states for
a hidden markov model
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculate the most likely sequence of hidden states for a
    hidden Markov model.

    Args:
        Observation (numpy.ndarray): An array of shape (T,) containing
        the index of observations.
        Emission (numpy.ndarray): An array of shape (N, M) containing
        emission probabilities.
        Transition (numpy.ndarray): An array of shape (N, N) containing
        transition probabilities.
        Initial (numpy.ndarray): An array of shape (N, 1) containing
        initial state probabilities.

    Returns:
        tuple: A tuple containing:
            - path (list): A list of length T with the most likely
            sequence of hidden states.
            - P (float): The probability of the path.
    """

    # Check input shapes and types
    if not isinstance(
            Observation,
            np.ndarray) or not isinstance(
            Emission,
            np.ndarray) or not isinstance(
                Transition,
                np.ndarray) or not isinstance(
                    Initial,
            np.ndarray):
        return None, None

    if Observation.ndim != 1 or Emission.ndim != 2 or \
            Transition.ndim != 2 or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape

    if Transition.shape != (N, N) or Initial.shape[1] != 1:
        return None, None

    T = Observation.shape[0]

    if T <= 0:
        return None, None

    # Check stochastic conditions
    if not np.isclose(
        np.sum(Initial),
        1) or not np.all(
        np.isclose(
            np.sum(
                Transition,
                axis=1),
            1)) or not np.all(
        np.isclose(
            np.sum(
                Emission,
                axis=1),
            1)):
        return None, None

    # Initialize V and B arrays
    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    # Initialize V[:, 0] using Initial and Emission
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Calculate V and B iteratively
    for t in range(1, T):
        for j in range(N):
            # Calculate the probabilities for all possible previous states
            probs = V[:, t - 1] * Transition[:, j] * \
                Emission[j, Observation[t]]
            # Store the maximum probability
            V[j, t] = np.max(probs)
            # Store the corresponding state that maximizes the probability
            B[j, t] = np.argmax(probs)

    # Backtrack to find the most likely path
    path = [np.argmax(V[:, T - 1])]
    for t in range(T - 1, 0, -1):
        path.append(B[path[-1], t])

    # Reverse the path to get the correct order
    path = path[::-1]

    # Calculate the probability of the path
    P = np.max(V[:, T - 1])

    return path, P

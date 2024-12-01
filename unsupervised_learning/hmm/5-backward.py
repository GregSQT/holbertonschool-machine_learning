#!/usr/bin/env python3
"""
the function def backward(Observation, Emission, Transition, Initial):
that performs the backward algorithm for a hidden markov model
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Perform the backward algorithm for a hidden Markov model.

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
            - P (float): The likelihood of the observations given the model.
            - B (numpy.ndarray): An array of shape (N, T) containing
              the backward path probabilities.
    """
    # Type and dimension checks
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    if Transition.shape != (N, N) or Initial.shape[1] != 1:
        return None, None

    # Stochastic checks
    if not np.sum(Emission, axis=1).all() or \
       not np.sum(Transition, axis=1).all() or \
       not np.sum(Initial) == 1:
        return None, None

    # Initialize Beta array
    B = np.zeros((N, T))

    # Initialization
    B[:, T - 1] = np.ones(N)

    # Recursion
    for t in range(T - 2, -1, -1):
        # Calculate the matrix multiplication of Transition, Emission,
        # and Beta at time t+1
        a = Transition
        b = Emission[:, Observation[t + 1]]
        c = B[:, t + 1]
        abc = a * b * c

        # Sum along the rows of abc to calculate the backward probabilities
        prob = np.sum(abc, axis=1)
        B[:, t] = prob

    # Calculate the likelihood of the observations
    P_first = Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0]
    P = np.sum(P_first)

    return P, B

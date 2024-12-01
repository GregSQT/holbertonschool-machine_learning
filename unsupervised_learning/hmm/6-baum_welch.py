#!/usr/bin/env python3
"""
the function def baum_welch(Observations, Transition, Emission, Initial,
iterations=1000):
that performs the Baum-Welch algorithm for a hidden markov model
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
    P = np.sum(F[:, -1])

    return P, F


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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden Markov model.

    Args:
        Observations (numpy.ndarray): An array of shape (T,) that contains
            the index of observations.
        Transition (numpy.ndarray): An array of shape (M, M) that contains
            the initialized transition probabilities.
        Emission (numpy.ndarray): An array of shape (M, N) that contains
            the initialized emission probabilities.
        Initial (numpy.ndarray): An array of shape (M, 1) that contains
            the initialized starting probabilities.
        iterations (int): The number of times expectation-maximization should
            be performed (default is 1000).

    Returns:
        tuple: A tuple containing the converged Transition and Emission
        matrices. Returns (None, None) on failure.
    """
    # Input Validation
    if not all(
        isinstance(
            arr, np.ndarray) and len(
            arr.shape) == 1 for arr in [Observations]):
        return None, None

    if not all(
        isinstance(
            arr,
            np.ndarray) and len(
            arr.shape) == 2 for arr in [
                Emission,
                Transition,
            Initial]):
        return None, None

    T = Observations.shape[0]
    N, M = Emission.shape

    a = Transition
    b = Emission
    a_prev = np.copy(a)
    b_prev = np.copy(b)

    for iteration in range(iterations):
        _, alpha = forward(Observations, b, a, Initial)
        _, beta = backward(Observations, b, a, Initial)

        xi = calculate_xi(Observations, alpha, beta, a, b)
        gamma = calculate_gamma(xi)

        a, b = update_parameters(Observations, xi, gamma, a, b)

        if np.all(np.isclose(a, a_prev)) or np.all(np.isclose(b, b_prev)):
            return a, b

        a_prev = np.copy(a)
        b_prev = np.copy(b)

    return a, b


def calculate_xi(Observations, alpha, beta, Transition, Emission):
    """
    Calculate the xi matrix for the Baum-Welch algorithm.

    Args:
        Observations (numpy.ndarray): An array of shape (T,) containing
            the index of observations.
        alpha (numpy.ndarray): The forward probabilities computed by the
            forward algorithm.
        beta (numpy.ndarray): The backward probabilities computed by the
            backward algorithm.
        Transition (numpy.ndarray): An array of shape (N, N) containing
            transition probabilities.
        Emission (numpy.ndarray): An array of shape (N, M) containing
            emission probabilities.

    Returns:
        numpy.ndarray: The xi matrix of shape (N, N, T-1) containing the
        joint probabilities of being in state i at time t and
        state j at time t+1
        given the observations and the model parameters.
    """
    # Calculate xi
    T = Observations.shape[0]
    N = alpha.shape[0]
    xi = np.zeros((N, N, T - 1))

    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                Fit = alpha[i, t]
                aij = Transition[i, j]
                bjt1 = Emission[j, Observations[t + 1]]
                Bjt1 = beta[j, t + 1]
                NUM = Fit * aij * bjt1 * Bjt1
                DEN = np.sum(alpha[:, t] * np.dot(Transition[:, j],
                             Emission[j, Observations[t + 1]]) *
                             beta[:, t + 1])
                xi[i, j, t] = NUM / DEN

    return xi


def calculate_gamma(xi):
    """
    Calculate the gamma matrix for the Baum-Welch algorithm.

    Args:
        xi (numpy.ndarray): The xi matrix of shape (N, N, T-1) containing
            the joint probabilities of being in state i at time t and state j
            at time t+1 given the observations and the model parameters.

    Returns:
        numpy.ndarray: The gamma matrix of shape (N, T) containing the
        probabilities of being in state i at time t given the observations
        and the model parameters.
    """
    # Calculate gamma
    gamma = np.sum(xi, axis=1)
    return gamma


def update_parameters(Observations, xi, gamma, Transition, Emission):
    """
    Update the model parameters (Transition and Emission) using the
    calculated xi and gamma matrices in the Baum-Welch algorithm.

    Args:
        Observations (numpy.ndarray): An array of shape (T,) that contains
            the index of observations.
        xi (numpy.ndarray): The xi matrix of shape (N, N, T-1) containing
            the joint probabilities of being in state i at time t and state j
            at time t+1 given the observations and the model parameters.
        gamma (numpy.ndarray): The gamma matrix of shape (N, T) containing
            the probabilities of being in state i at time t given the
            observations and the model parameters.
        Transition (numpy.ndarray): An array of shape (M, M) representing the
            transition probabilities of the hidden states.
        Emission (numpy.ndarray): An array of shape (M, N) representing the
            emission probabilities for each observation in each hidden state.

    Returns:
        tuple: A tuple containing the updated Transition and Emission matrices.

    Note:
        This function modifies the input Transition and Emission matrices
        in-place and returns the updated matrices.
    """

    # Update Transition and Emission matrices
    N = Transition.shape[0]
    M = Emission.shape[1]

    num = np.sum(xi, axis=2)
    den = np.sum(gamma, axis=1).reshape((-1, 1))
    new_transition = num / den

    xi_sum = np.sum(xi[:, :, -1], axis=0)
    xi_sum = xi_sum.reshape((-1, 1))
    gamma = np.hstack((gamma, xi_sum))

    denominator = np.sum(gamma, axis=1)
    denominator = denominator.reshape((-1, 1))

    new_emission = np.zeros((N, M))
    for i in range(M):
        gamma_i = gamma[:, Observations == i]
        new_emission[:, i] = np.sum(gamma_i, axis=1)

    new_emission = new_emission / denominator

    return new_transition, new_emission
    
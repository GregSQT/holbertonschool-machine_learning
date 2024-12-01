#!/usr/bin/env python3
"""Contains functions for Markov chains analysis."""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being
    in a particular state after a specified number of iterations.

    Args:
        P (numpy.ndarray): Transition matrix of shape (n, n),
            where P[i, j] is the probability of transitioning
            from state i to state j.
        s (numpy.ndarray): Initial probability distribution of shape (1, n),
            representing the probability of starting in each state.
        t (int): Number of iterations.

    Returns:
        numpy.ndarray: Probability distribution after t iterations,
        or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[1]:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None

    Pt = np.linalg.matrix_power(P, t)
    Ps = np.matmul(s, Pt)
    return Ps


def regular(P):
    """
    Determines if a Markov chain is regular.

    Args:
        P (numpy.ndarray): Transition matrix of shape (n, n),
            where P[i, j] is the probability of transitioning
            from state i to state j.

    Returns:
        bool: True if the Markov chain is regular, or False on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if P.shape != (n, n) or not np.allclose(np.sum(P, axis=1), np.ones(n)):
        return False
    if np.all(np.diag(P) == 1):
        return False
    return True


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Args:
        P (numpy.ndarray): Transition matrix of shape (n, n),
            where P[i, j] is the probability of transitioning
            from state i to state j.

    Returns:
        bool: True if the Markov chain is absorbing, or False on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if P.shape != (n, n) or not np.allclose(np.sum(P, axis=1), np.ones(n)):
        return False

    if np.all(np.diag(P) == 1):
        return True

    absorbing_states = np.where(np.diag(P) == 1)
    rows = P[absorbing_states[0]]
    account = np.sum(rows, axis=0)

    for i in range(n):
        row_check = P[i] != 0
        intersection = account * row_check

        if (intersection == 1).any():
            account[i] = 1

    return account.all()

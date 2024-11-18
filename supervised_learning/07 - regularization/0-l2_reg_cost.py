#!/usr/bin/env python3
"""
Exercice 0 : Regularization Cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Function that calculates cost of
    a neural network with L2 regularization"""

    frobenius_norm = 0  # Variable to store the Frobenius norm

    # Iterate over the weights dictionary to calculate the Frobenius norm
    for key, weight in weights.items():
        # Consider only the weight matrices,
        # identified by the key starting with 'W'
        if key[0] == 'W':
            # Calculate the norm of the weight matrix
            frobenius_norm += np.linalg.norm(weight)

    # Add the L2 regularization term to the cost
    cost += lambtha / (2 * m) * frobenius_norm

    return cost

#!/usr/bin/env python3
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Function that calculates the cost of a neural network with L2 regularization"""

    frobenius_norm = 0  # Variable to store the Frobenius norm

    # Iterate over the weights dictionary to calculate the Frobenius norm
    for key, weight in weights.items():
        if key[0] == 'W':  # Consider only the weight matrices, identified by the key starting with 'W'
            frobenius_norm += np.linalg.norm(weight)  # Calculate the norm of the weight matrix

    # Add the L2 regularization term to the cost
    cost += lambtha / (2 * m) * frobenius_norm

    return cost

if __name__ == '__main__':
    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['W2'] = np.random.randn(128, 256)
    weights['W3'] = np.random.randn(10, 128)

    cost = np.abs(np.random.randn(1))

    print(cost)
    cost = l2_reg_cost(cost, 0.1, weights, 3, 1000)
    print(cost)
    
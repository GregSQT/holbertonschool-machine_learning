#!/usr/bin/env python3
"""
Exercice 1 : Gradient Descent with L2 Regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function that updates the weights and biases of a neural network
    using gradient descent with L2 regularization.

    Arguments:
    - Y: numpy.ndarray of shape (output_size, m) representing the one-hot
         encoded labels for the training examples.
    - weights: dictionary of weights and biases for each layer of the network.
    - cache: dictionary of cached values (outputs of each layer) needed for
             computing the gradients.
    - alpha: learning rate for gradient descent.
    - lambtha: L2 regularization parameter.
    - L: number of layers in the neural network.

    Returns:
    - None (updates the weights in-place).

    """

    # Create a copy of the weights dictionary
    weights_copy = weights.copy()

    # Iterate over the layers of the neural network in reverse order
    for i in range(L, 0, -1):
        # Get the number of training examples
        m = Y.shape[1]

        # Calculate the derivative of activation function for current layer
        if i != L:
            # All layers use a tanh activation, except the last layer
            dZi = np.multiply(np.matmul(
                weights_copy['W' + str(i + 1)].T, dZi
            ), 1 - cache['A' + str(i)] ** 2)
        else:
            # Last layer uses a softmax activation
            dZi = cache['A' + str(i)] - Y

        # Calculate the gradients of the weights and biases
        dWi = np.matmul(dZi, cache['A' + str(i - 1)].T) / m
        dbi = np.sum(dZi, axis=1, keepdims=True) / m

        # Apply L2 regularization to the weights
        l2 = (1 - alpha * lambtha / m)
        weights['W' + str(i)] = l2 * weights_copy['W' + str(i)] - alpha * dWi
        weights['b' + str(i)] = weights_copy['b' + str(i)] - alpha * dbi

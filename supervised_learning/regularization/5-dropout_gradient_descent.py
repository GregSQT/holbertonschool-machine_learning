#!/usr/bin/env python3
"""
Exercice 5 : Gradient Descent with Dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights and biases of a neural
    network using gradient descent with Dropout.

    Arguments:
    - Y: The true labels of the training examples.
    - weights: Dictionary containing the weights and
    - biases of the neural network.
    - cache: Dictionary containing the cached
    - values from forward propagation.
    - alpha: Learning rate for gradient descent.
    - keep_prob: Probability of keeping
    - a neuron active during Dropout.
    - L: Number of layers in the neural network.

    Returns:
    - None
    """

    # Create a copy of the weights dictionary
    weights_copy = weights.copy()

    # Iterate over the layers in reverse order
    for i in range(L, 0, -1):
        m = Y.shape[1]

        if i != L:
            # Calculate dZi for hidden layer using tanh activation and Dropout
            dZi = np.multiply(np.matmul(
                weights_copy['W' + str(i + 1)].T, dZi
            ), tanh_prime(cache['A' + str(i)]))
            # Apply Dropout mask to dZi, regularize, normalize by keep_prob
            dZi *= cache['D' + str(i)]
            dZi /= keep_prob
        else:
            # Calculate dZi for the last layer using softmax activation
            dZi = cache['A' + str(i)] - Y

        # Calculate dWi and dbi for weight and bias updates
        dWi = np.matmul(dZi, cache['A' + str(i - 1)].T) / m
        dbi = np.sum(dZi, axis=1, keepdims=True) / m

        # Update weights and biases using gradient descent
        weights['W' + str(i)] = weights_copy['W' + str(i)] - alpha * dWi
        weights['b' + str(i)] = weights_copy['b' + str(i)] - alpha * dbi


def tanh(Y):
    """
    Define the hyperbolic tangent (tanh) activation function.

    Arguments:
    - Y: Input array.

    Returns:
    - Output of the tanh function.
    """
    return np.tanh(Y)


def tanh_prime(Y):
    """
    Define the derivative of the hyperbolic tangent (tanh) activation function

    Arguments:
    - Y: Input array.

    Returns:
    - Output of the derivative of the tanh function.
    """
    return 1 - Y ** 2

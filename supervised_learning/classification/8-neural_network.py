#!/usr/bin/env python3
"""Neural Network class"""
import numpy as np

# Define the NeuralNetwork class


class NeuralNetwork:
    """Class that defines a neural network with one hidden performing
    binary classification
    """

    def __init__(self, nx, nodes):
        """Class constructor"""
        # Validate and initialize the number of input features (nx)
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        # Validate and initialize the number of nodes in the hidden layer
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        # Initialize the weights and biases of the hidden layer
        self.W1 = np.random.randn(nodes, nx)  # Shape: (nodes, nx)
        self.b1 = np.zeros((nodes, 1))  # Shape: (nodes, 1)
        self.A1 = 0

        # Initialize the weights and biases of the output layer
        self.W2 = np.random.randn(1, nodes)  # Shape: (1, nodes)
        self.b2 = 0
        self.A2 = 0

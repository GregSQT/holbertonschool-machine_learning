#!/usr/bin/env python3
"""Deep Neural Network class"""
import numpy as np


class DeepNeuralNetwork:
    """Class that defines a neural network with one hidden performing
    binary classification
    """

    @staticmethod
    def he_et_al(nx, layers):
        """Calculates weights using he et al method"""
        weights = dict()
        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')
            prev_layer = layers[i - 1] if i > 0 else nx
            w_part1 = np.random.randn(layers[i], prev_layer)
            w_part2 = np.sqrt(2 / prev_layer)
            weights.update({
                'b' + str(i + 1): np.zeros((layers[i], 1)),
                'W' + str(i + 1): w_part1 * w_part2
            })
        return weights

    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = self.he_et_al(nx, layers)

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the deep neural network

        Args:
            X: input data

        Returns:
            Output of the neural network and the cache
        """
        self.cache.update({'A0': X})
        for i in range(self.L):
            A = self.cache.get('A' + str(i))
            biases = self.weights.get('b' + str(i + 1))
            weights = self.weights.get('W' + str(i + 1))
            Z = np.matmul(weights, A) + biases
            self.cache.update({'A' + str(i + 1): 1 / (1 + np.exp(-Z))})

        return self.cache.get('A' + str(i + 1)), self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                  np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data

        Returns:
            The neuron's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        return np.where(A <= 0.5, 0, 1), self.cost(Y, A)

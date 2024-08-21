#!/usr/bin/env python3
"""
Exercice 0. Neuron
"""

import numpy as np


class Neuron:
    """ Defines a single neuron performing binary classification """
    def __init__(self, nx):
        """
        Constructor method for Neuron instances.
        Args:
            nx : Number of input features to the neuron
                Must be an integer
                Must be positive
        Errors:
            TypeError: If nx is not an integer.
            ValueError: If nx less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        else:
            self.__W = np.random.normal(loc=0, scale=1, size=(1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron
        Args:
            X: input data
        Returns:
            Activation function - calculated with sigmoid function
        """
        A_prev = np.matmul(self.__W, X) + self.__b  # Perform matrix multiply
        self.__A = 1 / (1 + np.exp(-A_prev))  # Apply sigmoid activation func
        return self.__A

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
        """
        Evaluates the neuron's predictions
        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
        Returns:
            The neuron's prediction and the cost of the network
        """
        # Perform forward propagation to calculate the activation
        self.forward_prop(X)

        # Use a threshold of 0.5 to classify the predictions
        predictions = np.where(self.A <= 0.5, 0, 1)

        # Calculate the cost using the predicted values
        cost = self.cost(Y, self.A)

        # Return the predictions and the cost
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example
            alpha: learning rate
        """
        # Get the number of examples in the dataset
        m = Y.shape[1]

        # Calculate the difference between the activated output
        d_ay = A - Y

        # Calculate the gradient of the weights
        gradient = np.matmul(d_ay, X.T) / m

        # Calculate the gradient of the bias
        db = np.sum(d_ay) / m

        # Update the weights and bias using the gradient descent update rule
        self.__W -= gradient * alpha
        self.__b -= db * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains a neuron
        Args:
            X: contains the input data
            Y: contains the correct labels for the input data iteration:
               number of iterations to train over
            alpha: learning rate
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A, alpha)

        return self.evaluate(X, Y)

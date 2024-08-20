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
            self.W = np.random.normal(loc=0, scale=1, size=(1, nx))
            self.b = 0
            self.A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

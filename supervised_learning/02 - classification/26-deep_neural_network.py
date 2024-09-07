#!/usr/bin/env python3
"""Neural Network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Neural Network Class"""

    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1 or False in (np.array(layers) > 0):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                j = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights['W' + str(i + 1)] = j
            else:
                jjj = np.sqrt(2 / layers[i - 1])
                jj = np.random.randn(layers[i], layers[i - 1]) * jjj
                self.__weights['W' + str(i + 1)] = jj
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    def forward_prop(self, X):
        """Forward Propogation"""
        A = X
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            Z = np.matmul(W, A) + b
            A = self.sigmoid(Z)
            self.__cache['A' + str(i)] = A

        return A, self.__cache

    def sigmoid(self, X):
        """Sigmoid Helper"""
        return 1 / (1 + np.exp(-X))

    def cost(self, Y, A):
        """Cost Func"""
        m = Y.shape[1]
        j = np.log(1.0000001 - A)
        return ((-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * j))

    def evaluate(self, X, Y):
        """Evaluate Func"""
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient Descent"""
        m = Y.shape[1]
        L = self.__L

        A = cache["A" + str(L)]
        dZ = A - Y

        for li in range(L, 0, -1):
            A_prev = cache["A" + str(li - 1)]
            W = self.__weights["W" + str(li)]
            b = self.__weights["b" + str(li)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)

            self.__weights["W" + str(li)] -= alpha * dW
            self.__weights["b" + str(li)] -= alpha * db

            if li > 1:
                dZ = dA * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        Xgrp = []
        Ygrp = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

            if verbose:
                if i == 0 or i % step == 0:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, A)))

            if graph:
                if i == 0 or i % step == 0:
                    current_cost = self.cost(Y, A)
                    Ygrp.append(current_cost)
                    Xgrp.append(i)
                plt.plot(Xgrp, Ygrp)
                plt.xlabel('Iteration')
                plt.ylabel('Cost')
                plt.title('Training Cost')

            if verbose or graph:
                if not isinstance(step, int):
                    raise TypeError("step must be in integer")
                if step <= 0 or step > iterations:
                    raise ValueError("step must be positive and <= iterations")
        if graph:
            plt.show()
        return (self.evaluate(X, Y))

    def save(self, filename):
        """save a file"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        "load a file"
        try:
            if not filename.endswith('.pkl'):
                filename += '.pkl'

            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception:
            return None

    @property
    def L(self):
        """layer getter"""
        return self.__L

    @property
    def cache(self):
        '''itermed val getter'''
        return self.__cache

    @property
    def weights(self):
        '''weight getter'''
        return self.__weights

#!/usr/bin/env python3
"""
Exercice 2 : Shuffle Data
"""
import numpy as np


def shuffle_data(X, Y):
    """Function that shuffles the data points in two matrices the same way"""
    # Generate a random permutation of indices
    shuffle = np.random.permutation(X.shape[0])

    # Shuffle the rows of X based on the generated permutation
    X_shuf = X[shuffle]

    # Shuffle the rows of Y based on the generated permutation
    Y_shuf = Y[shuffle]
    return X_shuf, Y_shuf

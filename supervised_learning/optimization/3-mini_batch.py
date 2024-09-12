#!/usr/bin/env python3
"""
Exercice 3 : Trains a loaded neural network model 
             using mini-batch gradient descent
"""
import numpy as np


shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    '''
    Creates mini-batches to be used for training
    a neural network using mini-batch gradient descent.
    '''
    m = X.shape[0]
    mini_batches = []

    # Shuffle the data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    # Calculate the number of batches
    num_batches = m // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]

        mini_batches.append((X_batch, Y_batch))

    # Handle the last batch
    # (if the total number of data points is not divisible by batch_size)
    if m % batch_size != 0:
        start_idx = num_batches * batch_size
        X_batch = X_shuffled[start_idx:]
        Y_batch = Y_shuffled[start_idx:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches

#!/usr/bin/env python3

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    # Calculate the mean and variance along the first axis of Z
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    
    # Normalize Z using the mean and variance, adding epsilon for numerical stability
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    
    # Apply the scale and shift parameters
    out = gamma * Z_norm + beta
    return out


if __name__ == '__main__':
    np.random.seed(0)
    
    # Generate random data with specific means and standard deviations
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    
    # Concatenate the generated data along the second axis
    Z = np.concatenate((a, b, c), axis=1)
    
    # Generate random scale and shift parameters
    gamma = np.random.rand(1, 3)
    beta = np.random.rand(1, 3)
    
    # Set a small constant for numerical stability
    epsilon = 1e-8
    
    # Print the first 10 rows of the original data
    print(Z[:10])
    
    # Apply batch normalization to Z using the batch_norm function
    Z_norm = batch_norm(Z, gamma, beta, epsilon)
    
    # Print the first 10 rows of the normalized data
    print(Z_norm[:10])
    
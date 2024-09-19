#!/usr/bin/env python3

import numpy as np

# Import the normalization_constants function from the module '0-norm_constants'
normalization_constants = __import__('0-norm_constants').normalization_constants
# Import the normalize function from the module '1-normalize'
normalize = __import__('1-normalize').normalize


if __name__ == '__main__':
    np.random.seed(0)
    # Generate three arrays of random numbers
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    # Concatenate the arrays horizontally to form the dataset X
    X = np.concatenate((a, b, c), axis=1)
    
    # Calculate the normalization constants (mean and standard deviation) using the normalization_constants function
    m, s = normalization_constants(X)
    
    # Print the first 10 rows of the original dataset X
    print(X[:10])
    
    # Normalize the dataset X using the normalize function and the calculated normalization constants
    X = normalize(X, m, s)
    
    # Print the first 10 rows of the normalized dataset X
    print(X[:10])
    
    # Calculate the normalization constants of the normalized dataset
    m, s = normalization_constants(X)
    
    # Print the calculated mean and standard deviation values
    print(m)
    print(s)
    
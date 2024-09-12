#!/usr/bin/env python3

import numpy as np

# Import the shuffle_data function from the module '2-shuffle_data'
shuffle_data = __import__('2-shuffle_data').shuffle_data

if __name__ == '__main__':
    # Define the input arrays X and Y
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
    Y = np.array([[11, 12],
                  [13, 14],
                  [15, 16],
                  [17, 18],
                  [19, 20]])

    np.random.seed(0)
    # Shuffle the data using the shuffle_data function
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    # Print the shuffled X array
    print(X_shuffled)
    # Print the shuffled Y array
    print(Y_shuffled)
    
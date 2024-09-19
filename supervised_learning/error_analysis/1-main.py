#!/usr/bin/env python3

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity

if __name__ == '__main__':
    # Load the confusion matrix from the 'confusion.npz' file
    confusion = np.load('confusion.npz')['confusion']

    # Set print options to suppress scientific notation for readability
    np.set_printoptions(suppress=True)
    
    # Call the sensitivity function and print the result
    print(sensitivity(confusion))
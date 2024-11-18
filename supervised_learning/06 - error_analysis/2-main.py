#!/usr/bin/env python3
import numpy as np


# Import the precision function from the '2-precision' module
precision = __import__('2-precision').precision

if __name__ == '__main__':
    # Load the confusion matrix from the 'confusion.npz' file
    confusion = np.load('confusion.npz')['confusion']

    # Set the print options to suppress scientific notation
    np.set_printoptions(suppress=True)

    # Call the precision function with the loaded confusion matrix
    result = precision(confusion)

    # Print the resulting precision values
    print(result)
    
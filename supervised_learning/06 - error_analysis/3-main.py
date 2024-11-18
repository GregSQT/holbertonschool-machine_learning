#!/usr/bin/env python3
# Shebang line specifying the interpreter to be used when executing the script

import numpy as np
# Importing the numpy library and assigning it the alias 'np'


specificity = __import__('3-specificity').specificity
# Importing the 'specificity' function from the module '3-specificity'
# using the '__import__' function and assigning it to the variable 'specificity'

if __name__ == '__main__':
    # Check if the script is being run directly as the main module

    confusion = np.load('confusion.npz')['confusion']
    # Load the 'confusion' matrix from the file named 'confusion.npz'
    # using the 'np.load' function and indexing with ['confusion']
    # Assign the loaded matrix to the variable 'confusion'

    np.set_printoptions(suppress=True)
    # Set the print options for NumPy to suppress scientific notation

    print(specificity(confusion))
    # Call the 'specificity' function with the 'confusion' matrix as an argument
    # Print the result
    
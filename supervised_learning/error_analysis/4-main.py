#!/usr/bin/env python3
# Shebang line specifying the interpreter to be used when executing the script
import numpy as np
# Importing the numpy library and assigning it the alias 'np'


f1_score = __import__('4-f1_score').f1_score
# Importing the 'f1_score' function from the module '4-f1_score'
# using the '__import__' function and assigning it to the variable 'f1_score'

if __name__ == '__main__':
    # The following code block will be executed only if the script is run directly,
    # not if it is imported as a module in another script

    confusion = np.load('confusion.npz')['confusion']
    # Load the 'confusion' matrix from the 'confusion.npz' file using numpy's 'load' function
    # Access the 'confusion' matrix within the loaded data using square brackets
    # Assign the 'confusion' matrix to the variable 'confusion'

    np.set_printoptions(suppress=True)
    # Set numpy's print options to suppress the printing of very small numbers in scientific notation

    print(f1_score(confusion))
    # Call the 'f1_score' function with the 'confusion' matrix as an argument
    # Print the result to the console
    
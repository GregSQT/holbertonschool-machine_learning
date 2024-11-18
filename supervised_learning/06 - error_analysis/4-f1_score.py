#!/usr/bin/env python3
"""
Exercice 4 : F1 score
"""
import numpy as np
# Importing the numpy library and assigning it the alias 'np'


sensitivity = __import__('1-sensitivity').sensitivity
# Importing the 'sensitivity' function from the module '1-sensitivity'
# using the '__import__' function and assigning it to variable 'sensitivity'

precision = __import__('2-precision').precision
# Importing the 'precision' function from the module '2-precision'
# using the '__import__' function and assigning it to the variable 'precision'


def f1_score(confusion):
    """function that calculates the F1 score of a confusion matrix"""
    # Function definition with a docstring explaining its purpose

    s = sensitivity(confusion)
    # Call the 'sensitivity' function with 'confusion' matrix as an argument
    # Assign the result to the variable 's'

    p = precision(confusion)
    # Call the 'precision' function with the 'confusion' matrix as an argument
    # Assign the result to the variable 'p'

    f1 = 2 * (p * s) / (p + s)
    # Calculate the F1 score using the formulas:
    # 2 * (precision * sensitivity) / (precision + sensitivity)
    # Assign the result to the variable 'f1'

    return f1
    # Return the calculated F1 score

#!/usr/bin/env python3
"""
Exercice 2 : Precision
"""
import numpy as np


def precision(confusion):
    """Function that calculates the precision for each class."""

    # Get the number of classes from the shape of the confusion matrix
    classes, classes = confusion.shape

    # Create an array of zeros to store the precision for each class
    precision = np.zeros(shape=(classes,))

    # Iterate over each class
    for i in range(classes):
        # Calculate the precision for the current class
        # by dividing the true positives (confusion[i][i])
        # by the sum of true positives and false positives
        # for that class (np.sum(confusion, axis=0)[i])
        precision[i] = confusion[i][i] / np.sum(confusion, axis=0)[i]

    # Return the array of precision values
    return precision

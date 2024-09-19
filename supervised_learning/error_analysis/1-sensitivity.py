#!/usr/bin/env python3
"""
Exercice 1 : Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculate sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray):
        Confusion matrix of shape (classes, classes),
            where row indices represent the correct labels and
            column indices represent the predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,)
        containing the sensitivity of each class.
    """
    classes, classes = confusion.shape  # Get the number of classes
    # Initialize array to store sensitivity values
    sensitivity = np.zeros(shape=(classes,))

    for i in range(classes):
        # Calculate true positives for the current class
        true_positives = confusion[i][i]

        # Calculate actual positives for the current class
        actual_positives = np.sum(confusion, axis=1)[i]

        # Calculate sensitivity for the current class
        sensitivity[i] = true_positives / actual_positives

    return sensitivity

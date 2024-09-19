#!/usr/bin/env python3
"""
Exercice 3 : Specificity
"""
import numpy as np


def specificity(confusion):
    """function that calculates the specificity for each class"""
    classes, classes = confusion.shape  # Get the number of classes
    # Initialize an array to store specificity values
    specificity = np.zeros(shape=(classes,))
    for i in range(classes):
        # Calculate specificity for each class i
        # using the formula: (TN) / (TN + FP)
        # where TN is the number of true negatives
        # and FP is the number of false positives
        specificity[i] = (
            np.sum(confusion) - np.sum(confusion, axis=1)[i]
            - np.sum(confusion, axis=0)[i] + confusion[i][i]
        ) / (np.sum(confusion) - np.sum(confusion, axis=1)[i])
    return specificity

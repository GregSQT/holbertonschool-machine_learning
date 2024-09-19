#!/usr/bin/env python3
"""
Exercice 0 : Create Confusion
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """function that creates a confusion matrix"""
    # Get the shape of the labels array
    # (m: number of data points, classes: number of classes)
    m, classes = labels.shape

    # Find the indices of the maximum values in
    # each row of the labels and logits arrays
    v1 = np.argmax(labels, axis=1)
    v2 = np.argmax(logits, axis=1)

    # Create an empty confusion matrix with shape (classes, classes)
    confusion = np.zeros(shape=(classes, classes))

    # Iterate over each class combination and count
    # the occurrences in the data points
    for i in range(classes):
        for j in range(classes):
            for k in range(m):
                # Check if the current data point
                # has the correct label (i) and predicted label (j)
                if i == v1[k] and j == v2[k]:
                    # Increment the corresponding entry in confusion matrix
                    confusion[i][j] += 1

    return confusion

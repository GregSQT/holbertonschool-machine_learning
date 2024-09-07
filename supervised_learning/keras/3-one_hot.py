#!/usr/bin/env python3
"""
Exercice 3 : One Hot
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """function that converts a label vector into a one-hot matrix"""
    # Use the to_categorical function from Keras to convert labels into
    # one-hot encoding
    return K.utils.to_categorical(labels, num_classes=classes)

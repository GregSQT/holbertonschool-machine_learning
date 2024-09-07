#!/usr/bin/env python3
"""
Exercice 3 : Accuracy of a prediction
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction"""

    # Convert true labels and predicted labels to respective class indices
    label = tf.argmax(y, axis=1)
    pred = tf.argmax(y_pred, axis=1)

    # From tf .cast change boolean values to float (1 for True, 0 for False)
    # Calculate the mean accuracy over all predictions
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, label), tf.float32))

    return accuracy
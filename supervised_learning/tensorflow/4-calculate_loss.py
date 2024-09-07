#!/usr/bin/env python3
"""
Exercice 4 : Loss of a prediction
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """function that calculates the softmax cross-entropy loss of pred"""
    return tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

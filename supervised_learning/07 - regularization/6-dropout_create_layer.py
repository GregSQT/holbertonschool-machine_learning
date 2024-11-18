#!/usr/bin/env python3
"""
Exercice 6 : Create a Layer with L2 Dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.
    """
    # Weight initialization: He et. al
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    # Create Dense layer
    dense = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=init)

    # Apply dropout
    dropout = tf.nn.dropout(dense(prev), rate=1-keep_prob)

    return dropout

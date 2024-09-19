#!/usr/bin/env python3
"""
Exercice 6 : Create a Layer with L2 Dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """function that creates a tf layer using dropout"""

    # Initialize the weights of the layer using variance_scaling_initializer
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    # Create a dense layer with the specified number of neurons
    # activation function, initializer, and name
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )

    # Create a dropout layer with the specified keep probability
    drop = tf.layers.Dropout(rate=(1 - keep_prob))

    # Apply the dropout layer to the output of the dense layer
    return drop(layer(prev))

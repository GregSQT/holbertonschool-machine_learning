#!/usr/bin/env python3
"""
Exercice 1 - Layers
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Create a layer in a neural network.

    Arguments:
    prev : tensor output of the previous layer
    n : number of nodes in the layer to create
    activation : activation function to be used

    Returns:
    Tensor output of the layer
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name="layer")
    return layer(prev)
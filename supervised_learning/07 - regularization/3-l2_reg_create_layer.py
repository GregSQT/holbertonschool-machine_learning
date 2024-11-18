#!/usr/bin/env python3
"""
Exercice 3 : Create a Layer with L2 Regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a dense layer with L2 regularization."""
    init_weights = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                         mode="fan_avg")
    l2_regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init_weights,
        kernel_regularizer=l2_regularizer
    )

    return layer(prev)

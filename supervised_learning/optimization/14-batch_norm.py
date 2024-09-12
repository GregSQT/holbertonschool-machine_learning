#!/usr/bin/env python3
"""
Exercice 14 : Batch Normalization Upgraded
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev: The activated output of the previous layer.
        n: The number of nodes in the layer to be created.
        activation: The activation function to be used
        on the output of the layer.

    Returns:
        A tensor of the activated output for the layer.
    """

    # Initialize the layer with the variance scaling initializer
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    # Create a dense layer with the specified number of units, no activation
    # function
    layer = tf.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=initializer,
        name='layer')

    # Pass the previous layer through the dense layer to obtain the layer
    # output
    layer_output = layer(prev)

    # Calculate the mean and variance of the layer output
    mean, variance = tf.nn.moments(layer_output, axes=[0])

    # Create trainable variables for beta and gamma, initialized as vectors of
    # zeros and ones.
    beta = tf.Variable(
        tf.zeros(
            shape=(
                1,
                n),
            dtype=tf.float32),
        trainable=True,
        name='beta')
    gamma = tf.Variable(
        tf.ones(
            shape=(
                1,
                n),
            dtype=tf.float32),
        trainable=True,
        name='gamma')

    # Set a small constant epsilon for numerical stability
    epsilon = 1e-08

    # Apply batch normalization to the layer output using the calculated mean,
    # variance, beta, gamma, and epsilon
    Z_b_norm = tf.nn.batch_normalization(
        x=layer_output,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon,
        name=None)

    # Apply the specified activation function if provided
    if activation:
        return activation(Z_b_norm)

    # Return the batch-normalized output if no activation function is provided
    return Z_b_norm

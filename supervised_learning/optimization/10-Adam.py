#!/usr/bin/env python3
"""
Exercice 10 : Adam Upgraded
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Function that implements Adam gradient descent in TensorFlow.

    Args:
        loss: The loss of the network.
        alpha: The learning rate, controls the step size during optimization
        beta1: The weight used for the first moment estimation in Adam.
        beta2: The weight used for the second moment estimation in Adam.
        epsilon: A small number to avoid division by zero.

    Returns:
        The Adam optimization operation.
    """
    return tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon,
        use_locking=False, name='Adam'
    ).minimize(loss)
    
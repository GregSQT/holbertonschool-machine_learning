#!/usr/bin/env python3
"""
Exercice 5 : Train_Op
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Function that creates the training operation for the network.

    Args:
        loss:  The loss value of the network's prediction.
        alpha: The learning rate or step size for
               the gradient descent optimizer.

    Returns:
        The training operation that minimizes the loss.
    """

    # A TensorFlow gradient descent optimizer with the specified learning rate
    optimizer = tf.train.GradientDescentOptimizer(alpha)

    # Create the training operation that minimizes the loss
    train_op = optimizer.minimize(loss)

    # Return the training operation
    return train_op

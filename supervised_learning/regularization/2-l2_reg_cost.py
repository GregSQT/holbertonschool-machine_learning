#!/usr/bin/env python3
"""
Exercice 2 : L2 Regularization Cost
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Function that calculates the cost of a neural network
    with L2 regularization.

    Arguments:
    - cost: TensorFlowscalar representing original cost without regularization

    Returns:
    - TensorFlow scalar representing the cost with L2 regularization.
    """
    return cost + tf.losses.get_regularization_losses()

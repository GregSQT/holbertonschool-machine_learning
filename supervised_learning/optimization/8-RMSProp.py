#!/usr/bin/env python3
"""
Exercice 8 : RMSProp Upgraded
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Create the RMSProp optimization operation in TensorFlow.

    Args:
        loss: Loss of the network.
        alpha: Learning rate.
        beta2: RMSProp weight.
        epsilon: Small number to avoid division by zero.

    Returns:
        RMSProp optimization operation.
    """
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha, decay=beta2, momentum=0.0, epsilon=epsilon,
        use_locking=False, centered=False, name='RMSProp'
    )
    train_op = optimizer.minimize(loss)
    return train_op

#!/usr/bin/env python3
"""
Exercice 6 : Momentum Upgraded
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """function that implements momentum gradient descent in tensorflow"""
    return tf.train.MomentumOptimizer(
        # Set the learning rate, momentum, and locking parameters
        learning_rate=alpha, momentum=beta1, use_locking=False,
        # Specify the optimizer name and whether to use Nesterov momentum
        name='Momentum', use_nesterov=False
    ).minimize(loss)  # Minimize the provided loss using Momentum optimizer
    
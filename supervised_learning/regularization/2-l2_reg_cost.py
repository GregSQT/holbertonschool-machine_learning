#!/usr/bin/env python3
"""
Exercice 2 : L2 Regularization Cost
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """ calculates the cost of a neural network with L2 regularization"""

    l2_losses = []
    for layer in model.layers:
        if layer.losses:
            l2_loss = tf.reduce_sum(layer.losses)
            l2_losses.append(l2_loss)
    l2_losses_tensor = tf.convert_to_tensor(l2_losses)
    total_costs = cost + l2_losses_tensor

    return total_costs

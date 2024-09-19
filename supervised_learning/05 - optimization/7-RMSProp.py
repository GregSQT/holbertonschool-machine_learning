#!/usr/bin/env python3
"""
Exercice è : RMSProp
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update a variable using the RMSProp optimization algorithm.

    Args:
        alpha: The learning rate.
        beta2: The RMSProp weight.
        epsilon: A small number to avoid division by zero.
        var: The variable to be updated.
        grad: The gradient of var.
        s: The previous second moment of var.

    Returns:
        The updated variable and the new moment, respectively.
    """
    # Calculate the updated second moment using RMSProp formula
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update the variable using the learning rate, second moment, and gradient
    var -= alpha * (grad / (np.sqrt(s) + epsilon))

    return var, s

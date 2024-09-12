#!/usr/bin/env python3
"""
Exercice 9 : Adam
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update a variable in place using the Adam optimization algorithm.

    Args:
        alpha: Learning rate.
        beta1: Weight used for the first moment.
        beta2: Weight used for the second moment.
        epsilon: Small number to avoid division by zero.
        var: Numpy array containing the variable to be updated.
        grad: Numpy array containing the gradient of var.
        v: Previous first moment of var.
        s: Previous second moment of var.
        t: Time step used for bias correction.

    Returns:
        Updated variable, new first moment, and new second moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    # Introduce bias correction for the first moment
    v_corr = v / (1 - (beta1 ** t))

    s = beta2 * s + (1 - beta2) * (grad ** 2)
    # Introduce bias correction for the second moment
    s_corr = s / (1 - (beta2 ** t))
    # Update variable using Adam update rule
    var -= alpha * (v_corr / (np.sqrt(s_corr) + epsilon))

    return var, v, s

#!/usr/bin/env python3
"""
Exercice 5 : Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Function that updates a variable using a
    gradient descent with momentum optimization.

    Args:
        alpha: The learning rate.
        beta1: The momentum weight.
        var: The variable to be updated.
        grad: The gradient of var.
        v: The previous first moment of var.

    Returns:
        The updated variable and the new moment, respectively.
    """
    # Update the first moment using the momentum formula
    v = beta1 * v + (1 - beta1) * grad

    # Update the variable using the learning rate and the new first moment
    var -= alpha * v

    return var, v

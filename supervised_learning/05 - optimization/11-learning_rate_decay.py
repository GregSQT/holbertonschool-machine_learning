#!/usr/bin/env python3
"""
Exercice 11 : Learning Rate Decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that updates the learning rate using inverse time decay.

    Args:
        alpha: The original learning rate.
        decay_rate: The weight used to determine
        the rate at which alpha will decay.
        global_step: The number of passes of gradient descent have elapsed.
        decay_step: The number of passes of gradient descent that
        should occur before alpha is decayed further.

        decay_rate: The weight used to determine
        the rate at which alpha will decay.
        global_step: The number of passes of gradient descent have elapsed.
        decay_step: The number of passes of gradient descent
        that should occur before alpha is decayed further.

    Returns:
        The updated value for alpha.
    """

    # Calculate the updated learning rate using inverse time decay formula
    alpha = alpha / (1 + decay_rate * int(global_step / decay_step))

    # Return the updated learning rate
    return alpha

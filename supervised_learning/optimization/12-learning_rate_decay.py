#!/usr/bin/env python3
"""
Exercice 12 : Learning Rate Decay Upgraded
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Function that updates the learning rate using inverse time decay"""

    # Create a learning rate decay operation using inverse time decay
    # The learning rate will decay over time based on the specified parameters
    # The decay occurs in a stepwise fashion if
    # the "staircase" parameter is set to True
    # The operation name is set to None by default
    learning_rate = tf.train.inverse_time_decay(
        learning_rate=alpha, global_step=global_step, decay_steps=decay_step,
        decay_rate=decay_rate, staircase=True, name=None
    )

    # Return the learning rate decay operation
    return learning_rate

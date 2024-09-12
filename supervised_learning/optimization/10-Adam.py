#!/usr/bin/env python3
"""
Exercice 10 : Adam Upgraded
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''
    Method that creates training op for NN
    in tf using Adam optimization algo
    '''

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    train_op = optimizer.minimize(loss)

    return train_op

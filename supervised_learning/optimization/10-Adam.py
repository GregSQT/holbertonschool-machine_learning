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

    a = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return a.minimize(loss)

#!/usr/bin/env python3
"""
Exercice 0 : Placeholders
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for input datas and labels.

    Arguments:
    nx -- number of featured columns in the datas
    classes -- number of classes in the classifier

    Returns:
    x : placeholder for input datas
    y : placeholder for one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y

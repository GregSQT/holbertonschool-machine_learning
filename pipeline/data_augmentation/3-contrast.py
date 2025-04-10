#!/usr/bin/env python3
"""
Function to randomly adjust the contrast of an image
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Contrast of an image randomly adjusted
    """
    return tf.image.random_contrast(image, lower, upper)

#!/usr/bin/env python3
"""
Function to rotate an image by 90 degrees counter-clockwise
"""
import tensorflow as tf


def rotate_image(image):
    """
    90 degrees counter-clockwise image rotation
    """
    return tf.image.rot90(image)

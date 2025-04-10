#!/usr/bin/env python3
"""
Function that flips an image horizontally
"""
import tensorflow as tf


def flip_image(image):
    """
    Horizontal image flip
    """
    return tf.image.flip_left_right(image)

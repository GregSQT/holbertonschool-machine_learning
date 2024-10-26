#!/usr/bin/env python3
"""
Neural Style Transfer (NST) class.
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Performs Neural Style Transfer tasks.

    Attributes:
        style_layers (list): Layers used for extracting style features.
        content_layer (str): Layer used for extracting content features.
        style_image: The preprocessed style image.
        content_image: The preprocessed content image.
        alpha (float): Weight for content cost.
        beta (float): Weight for style cost.
    """

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes NST with style and content images and their weights.

        Args:
            style_image (np.ndarray): Image used as a style reference.
            content_image (np.ndarray): Image used as a content reference.
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.

        Raises:
            TypeError: If style_image is not a valid image.
            TypeError: If content_image is not a valid image.
            TypeError: If alpha or beta are not non-negative numbers.
        """
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 \
           or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image to have pixel values between 0 and 1 and
        the largest side as 512 pixels.

        Args:
            image (np.ndarray): Image to be scaled.

        Returns:
            A scaled image with a new shape where the largest dimension
            is 512 pixels.

        Raises:
            TypeError: If image is not a valid image.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 \
           or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        max_dim = 512
        scale = max_dim / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        image = tf.image.resize(
            image, [new_h, new_w], method=tf.image.ResizeMethod.BICUBIC
        )

        image = image / 255.0
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        return image

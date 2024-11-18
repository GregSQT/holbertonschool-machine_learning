#!/usr/bin/env python3
"""
Neural Style Transfer (NST) class that performs tasks for neural
style transfer.
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Neural Style Transfer (NST) class that performs neural style
    transfer tasks.

    Attributes:
        style_layers (list): Layers used for extracting style features.
        content_layer (str): Layer used for extracting content features.
        style_image (tf.Tensor): Preprocessed style image.
        content_image (tf.Tensor): Preprocessed content image.
        alpha (float): Weight for content cost.
        beta (float): Weight for style cost.
        model (tf.keras.Model): Keras model used to calculate the costs.
    """

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class with style and content images, their weights,
        and loads the model used for neural style transfer.

        Args:
            style_image (np.ndarray): The image used as a style reference,
                                      expected shape (h, w, 3).
            content_image (np.ndarray): The image used as a content reference,
                                        expected shape (h, w, 3).
            alpha (float): The weight for content cost. Defaults to 1e4.
            beta (float): The weight for style cost. Defaults to 1.

        Raises:
            TypeError: If style_image is not a np.ndarray of shape (h, w, 3).
            TypeError: If content_image is not a np.ndarray of shape (h, w, 3).
            TypeError: If alpha or beta are not non-negative numbers.
        """
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 \
           or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(
            content_image, np.ndarray) or content_image.ndim != 3 \
           or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Preprocess images
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image (np.ndarray) The image to be scaled, expected shape (h, w, 3)

        Returns:
            A scaled image with shape (1, h_new, w_new, 3)
            where max(h_new, w_new)
            == 512 and the pixel values are in range [0, 1].

        Raises:
            TypeError: If image is not a np.ndarray with shape (h, w, 3).
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

    def load_model(self):
        """
        Creates the model used to calculate cost from VGG19 Keras base model

        Model's input should match VGG19 input
        Model's output should be a list containing outputs of VGG19 layers
            listed in style_layers followed by content_layers

        Saves the model in the instance attribute model
        """
        # Load the VGG19 model pre-trained on ImageNet
        VGG19_model = tf.keras.applications.VGG19(include_top=False,
                                                  weights='imagenet')

        # Save and reload the model with custom objects
        VGG19_model.save("VGG19_base_model")
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}

        # Load the model with custom objects
        vgg = tf.keras.models.load_model("VGG19_base_model",
                                         custom_objects=custom_objects)

        # Initialize lists for outputs
        style_outputs = []
        content_output = None

        # Iterate through the layers of the model & collect the desired outputs
        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name == self.content_layer:
                content_output = layer.output

            # Freeze the layer to make it non-trainable
            layer.trainable = False

        # Combine the style and content outputs
        outputs = style_outputs + [content_output]

        # Create the model that outputs the desired layers
        model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

        # Save the model in the instance attribute
        self.model = model

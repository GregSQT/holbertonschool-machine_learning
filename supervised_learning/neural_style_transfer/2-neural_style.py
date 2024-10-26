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
        Loads the VGG19 model and modifies it to output the activations of the
        style and content layers required for neural style transfer.

        The model's input will remain the same as the original VGG19, but the
        output will include the activations from the layers specified in the
        style_layers and content_layer attributes.

        Returns:
            The Keras model with the selected style and content layer outputs.
        """
        # Load the VGG19 model pre-trained on ImageNet
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')

        # Freeze the layers
        vgg.trainable = False

        # Get outputs of the selected style and content layers
        style_outputs = (
            [vgg.get_layer(layer).output for layer in self.style_layers])
        content_output = vgg.get_layer(self.content_layer).output

        # Model that takes VGG19 inputs and outputs style and content layers
        model_outputs = style_outputs + [content_output]
        model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)

        return model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix for a given input tensor.

        Args:
            input_layer (tf.Tensor or tf.Variable): A tensor of shape
                                                    (1, h, w, c) containing
                                                    the layer output whose
                                                    gram matrix should be
                                                    calculated.

        Raises:
            TypeError: If input_layer is not a tensor of rank 4.

        Returns:
            tf.Tensor: of shape (1, c, c) representing the gram matrix.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Unpack the shape of the input layer
        batch, height, width, channels = input_layer.shape

        # Reshape the input to (height * width, channels)
        reshaped_input = tf.reshape(input_layer, (height * width, channels))

        # Calculate the Gram matrix
        gram = tf.matmul(reshaped_input, reshaped_input, transpose_a=True)

        # Normalize the Gram matrix by dividing by the number of elements
        gram_matrix = gram / tf.cast(height * width, tf.float32)

        # Reshape the output to (1, c, c)
        return tf.expand_dims(gram_matrix, axis=0)

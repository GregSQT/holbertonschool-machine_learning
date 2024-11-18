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
        self.generate_features()

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

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.

        Sets the public instance attributes:
            gram_style_features and content_feature
        """
        vgg = tf.keras.applications.vgg19

        # Preprocess the style and content images
        preprocess_style = vgg.preprocess_input(self.style_image * 255)
        preprocess_content = vgg.preprocess_input(self.content_image * 255)

        # Extract features from the preprocessed images
        style_features = self.model(preprocess_style)[:-1]
        content_feature = self.model(preprocess_content)[-1]

        # Calculate Gram matrices for the style features
        gram_style_features = (
            [self.gram_matrix(feature) for feature in style_features])

        # Set the instance attributes
        self.gram_style_features = gram_style_features
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.

        Args:
            style_output (tf.Tensor): Tensor of shape (1, h, w, c) containing
                                the layer style output of the generated image.
            gram_target (tf.Tensor): Tensor of shape (1, c, c) containing the
                                    gram matrix of the target style output for
                                    that layer.

        Raises:
            TypeError: If style_output is not a tf.Tensor or
            tf.Variable of rank 4.
            TypeError: If gram_target is not a tf.Tensor or
            tf.Variable with shape
                (1, c, c), where c is the number of channels in style_output.

        Returns:
            tf.Tensor: The layer's style cost.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                gram_target.shape != (1, style_output.shape[-1],
                                      style_output.shape[-1]):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, "
                f"{style_output.shape[-1]}, {style_output.shape[-1]}]")

        # Calculate the Gram matrix for the style output
        gram_style_output = self.gram_matrix(style_output)

        # Compute the style cost (mean squared error between the Gram matrices)
        style_cost = tf.reduce_mean(tf.square(gram_style_output - gram_target))

        return style_cost

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image.

        Args:
            style_outputs (list): A list of tf.Tensor style outputs for the
                                generated image.

        Raises:
            TypeError: If style_outputs is not a list with the same length
                    as self.style_layers.

        Returns:
            tf.Tensor: The style cost for the generated image.
        """
        if not isinstance(style_outputs, list) or \
                len(style_outputs) != len(self.style_layers):
            raise TypeError(
                f"style_outputs must be a list with a length of "
                f"{len(self.style_layers)}"
            )

        # Initialize the style cost
        total_style_cost = 0.0

        # Number of style layers
        weight_per_layer = 1.0 / len(self.style_layers)

        # Loop through each style output and corresponding gram target
        for style_output, gram_target in zip(style_outputs,
                                             self.gram_style_features):
            # Calculate the style cost for each layer using layer_style_cost
            layer_cost = self.layer_style_cost(style_output, gram_target)
            # Weight the layer cost and add it to the total style cost
            total_style_cost += weight_per_layer * layer_cost

        return total_style_cost

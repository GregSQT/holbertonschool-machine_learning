#!/usr/bin/env python3
""" This module defines the class NST that performs Neural Style Transfer """
import numpy as np
import tensorflow as tf


class NST:
    """ This class performs tasks for Neural Style Transfer """
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST class that initializes the following public
        instance attributes:
        - style_image (numpy.ndarray): the image used as a style reference.
        - content_image (numpy.ndarray): the image used as a content reference.
        - alpha (float): the weight for content cost.
        - beta (float): the weight for style cost.
        """
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if (not isinstance(style_image, np.ndarray) or style_image.ndim != 3
                or style_image.shape[-1] != 3):
            raise TypeError(error1)
        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3 or content_image.shape[-1] != 3):
            raise TypeError(error2)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        # Load the model
        self.load_model()
        # Generate the features
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        This method rescales the image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels.
        Args:
            image (numpy.ndarray): the image to be rescaled.
        """
        error = "image must be a numpy.ndarray with shape (h, w, 3)"
        if (not isinstance(image, np.ndarray) or len(image.shape) != 3
                or image.shape[-1] != 3):
            raise TypeError(error)
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * h_new / h)
        else:
            w_new = 512
            h_new = int(h * w_new / w)
        # Resize the image with bicubic interpolation
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize(image, (h_new, w_new), method="bicubic")
        # Normalize the image pixel values to be in the range [0, 1]
        image = tf.clip_by_value(image / 255.0, 0, 1)
        return image[tf.newaxis, ...]

    def load_model(self):
        """
        This method loads the VGG19 model for Neural Style Transfer.
        It returns the model.
        """
        # Load the VGG19 model
        base_vgg = tf.keras.applications.VGG19(include_top=False,
                                               weights='imagenet')
        # Replace MaxPooling layers with Average Pooling
        # Achieved by utilizing the custom_objects parameter
        # during model loading
        custom_object = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        base_vgg.save("base_vgg")
        # Reload the VGG model with the pooling layers swapped
        vgg = tf.keras.models.load_model("base_vgg",
                                         custom_objects=custom_object)
        # Make sure that the model is non-trainable
        vgg.trainable = False
        # Get output layers corresponding to style and content layers
        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        # Create a model that returns the outputs of the VGG19 model
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        This method calculates the gram matrices for the input layer.
        It returns the gram matrix.
        Args:
            input_layer (tf.Tensor): the layer from which to calculate the gram
            matrix.
        Returns:
            A tf.Tensor of shape (1, c, c) containing the gram matrix of the
            input layer.
        """
        error = "input_layer must be a tensor of rank 4"
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError(error)

        _, h, w, c = input_layer.shape
        # Reshape the features of the layer to a 2D matrix
        F = tf.reshape(input_layer, (h * w, c))
        # Calculate the gram matrix
        gram = tf.matmul(F, F, transpose_a=True)
        # Expand dimensions to have shape (1, c, c)
        gram = tf.expand_dims(gram, axis=0)

        # Normalize by number of locations (h * w) then return gram tensor
        nb_locations = tf.cast(h * w, tf.float32)

        return gram / nb_locations

    def generate_features(self):
        """
        This method extracts the features used to calculate Neural Style
        Transfer cost from the content and style images.
        It returns the style features and content feature.
        Args:
            None
        Returns:
            A list of tf.Tensor objects containing the style features and the
            content feature.
        """
        # Preprocess the images
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        # Get the style features and content features
        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)
        # Calculate the gram matrices for the style features
        self.gram_style_features = [self.gram_matrix(style_feature)
                                    for style_feature in style_outputs[:-1]]
        # Get the content feature
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        This method calculates the style cost for a single layer.
        Args:
            style_output (tf.Tensor): the layer's style output of shape
                (1, h, w, c).
            gram_target (tf.Tensor): the style target's gram matrix for that
            layer of shape (1, c, c).
        Returns:
            The layer's style cost.
        """
        error = "style_output must be a tensor of rank 4"
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError(error)
        # last dimension of style_output
        c = style_output.shape[-1]
        error = f"gram_target must be a tensor of shape [1, {c}, {c}]"
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != (1, c, c)):
            raise TypeError(error)
        # Calculate the style cost
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        This method calculates the style cost for the generated image.
        Args:
            style_outputs (list of tf.Tensor): the style outputs for the
                generated image.
        Returns:
            The style cost.
        """
        len_style_layers = len(self.style_layers)
        error = f"style_outputs must be a list with a length of \
{len_style_layers}"
        if (not isinstance(style_outputs, list)
                or len(style_outputs) != len_style_layers):
            raise TypeError(error)
        style_cost = 0
        for target, output in zip(self.gram_style_features, style_outputs):
            style_cost += self.layer_style_cost(output, target)
        return style_cost / len_style_layers

    def content_cost(self, content_output):
        """
        This method calculates the content cost for the generated image.
        Args:
            content_output (tf.Tensor): the content output for the generated
                image.
        Returns:
            The content cost.
        """
        s = self.content_feature.shape
        error = f"content_output must be a tensor of shape {s}"
        if (not isinstance(content_output, (tf.Tensor, tf.Variable))
                or content_output.shape != s):
            raise TypeError(error)
        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        This method calculates the total cost for the generated image.
        Args:
            generated_image (tf.Tensor): a Tensor of shape (1, nh, nw, 3)
                containing the generated image.
        Returns:
            (J, J_content, J_style) where:
                J is the total cost.
                J_content is the content cost.
                J_style is the style cost.
        """
        s = self.content_image.shape
        error = f"generated_image must be a tensor of shape {s}"
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != s):
            raise TypeError(error)
        # Preprocess the generated image
        generated_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        # Get the outputs of the generated image
        outputs = self.model(generated_image)
        # Separate the style outputs and content output
        style_outputs = outputs[:-1]
        content_output = outputs[-1]
        # Calculate the style cost
        style_cost = self.style_cost(style_outputs)
        # Calculate the content cost
        content_cost = self.content_cost(content_output)
        # Calculate the total cost
        total_cost = self.alpha * content_cost + self.beta * style_cost
        return total_cost, content_cost, style_cost

    def compute_grads(self, generated_image):
        """
        This method computes the gradients for the generated image.
        Args:
            generated_image (tf.Tensor): a tf.Tensor of shape (1, nh, nw, 3)
                containing the generated image.
        Returns:
            (gradients, J_total, J_content, J_style) where:
                gradients is a tf.Tensor containing the gradients for the
                    generated image.
                J_total is the total cost for the generated image.
                J_content is the content cost for the generated image.
                J_style is the style cost for the generated image.
        """
        s = self.content_image.shape
        error = f"generated_image must be a tensor of shape {s}"
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != s):
            raise TypeError(error)
        # Compute the gradients
        with tf.GradientTape() as tape:
            # Get the total cost
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)
        # Get the gradients
        gradients = tape.gradient(J_total, generated_image)
        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """
        This method generates the Neural Style Transfer image.
        Args:
        iterations (int): the number of iterations to perform gradient
            descent over.
        step (int): the step at which to print information about the
            training, including the final iteration:
            print: Cost at iteration {i}: {J_total}, content
            {J_content}, style {J_style}
                i is the iteration
                J_total is the total cost
                J_content is the content cost
                J_style is the style cost
        lr (float): the learning rate for gradient descent.
        beta1 (float): the beta1 parameter for gradient descent.
        beta2 (float): the beta2 parameter for gradient descent.
        Returns:
        (generated_image, cost) where:
            generated_image is the generated image.
            cost is the cost of the generated image.
        """
        error = "iterations must be an integer"
        if not isinstance(iterations, int):
            raise TypeError(error)
        error = "iterations must be positive"
        if isinstance(iterations, int) and iterations <= 0:
            raise TypeError(error)
        error = "step must be an integer"
        if step is None or not isinstance(step, int):
            raise TypeError(error)
        error = "step must be positive and less than iterations"
        if (step is not None and isinstance(step, int)
                and not (step > 0 or step < iterations)):
            raise TypeError(error)
        error = "lr must be a number"
        if not isinstance(lr, (int, float)):
            raise TypeError(error)
        error = "lr must be positive"
        if isinstance(lr, (int, float)) and lr <= 0:
            raise TypeError(error)
        error = "beta1 must be a float"
        if not isinstance(beta1, float):
            raise TypeError(error)
        error = "beta1 must be in the range [0, 1]"
        if isinstance(beta1, float) and (beta1 < 0 or beta1 > 1):
            raise TypeError(error)
        error = "beta2 must be a float"
        if not isinstance(beta2, float):
            raise TypeError(error)
        error = "beta2 must be in the range [0, 1]"
        if isinstance(beta2, float) and (beta2 < 0 or beta2 > 1):
            raise TypeError(error)

        # Initialize the Adam optimizer
        optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta1,
                                       beta_2=beta2)
        # Initialize the generated image to be a copy of the content image
        generated_image = tf.Variable(self.content_image)
        # Perform optimization
        best_cost = float('inf')
        best_image = None

        for i in range(iterations):
            with tf.GradientTape() as tape:
                grads, J_total, J_content, J_style = self.compute_grads(
                    generated_image)

            # Applying gradients to the generated image
            optimizer.apply_gradients([(grads, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, 0, 1))

            # Print each `step`
            if step is not None and i % step == 0:
                print(f"Cost at iteration {i}: {J_total.numpy()}, \
content {J_content.numpy()}, style {J_style.numpy()}")

            # Save the best image
            if J_total < best_cost:
                best_cost = J_total
                prev_image = generated_image
        # Removes the extra dimension from the image
        best_image = prev_image[0].numpy()
        return best_image, best_cost

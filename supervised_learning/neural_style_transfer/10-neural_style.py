#!/usr/bin/env python3
"""Neural Style Transfer project by Mateo Quirós Asprón"""
import numpy as np
import tensorflow as tf


class NST:
    """Neural style transfer class"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """Initializes the class"""
        if not isinstance(style_image, np.ndarray):
            raise TypeError("style_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray):
            raise TypeError("content_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("alpha must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales the image"""
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray \
with shape (h, w, 3)")

        height, width = image.shape[:2]
        max_side = np.maximum(height, width)
        scaling_factor = 512/max_side
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        rescaled_image = tf.image.resize(image, (new_height,
                                         new_width), method='bicubic')
        rescaled_image = rescaled_image / 255.0
        rescaled_image = tf.clip_by_value(rescaled_image, 0, 1)
        return rescaled_image

    def load_model(self):
        """Loads the model that will be used for feature extraction"""
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                weights='imagenet')
        x = vgg.input
        model_outputs = []
        content_output = None
        for layer in vgg.layers[1::]:
            if "pool" in layer.name:
                x = tf.keras.layers.AveragePooling2D(pool_size=layer.pool_size,
                                                     strides=layer.strides,
                                                     name=layer.name)(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    model_outputs.append(x)
                if layer.name == self.content_layer:
                    content_output = x
                layer.trainable = False
        model_outputs.append(content_output)
        model = tf.keras.models.Model(vgg.input, model_outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates gram matrices"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if tf.rank(input_layer) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        _, h, w, c = input_layer.shape
        # Flatten spatial dimensions
        flattened_layer = tf.reshape(input_layer, [1, h*w, c])
        # Compute Gram matrix
        gram = tf.matmul(flattened_layer, flattened_layer, transpose_a=True)
        # Normalize by number of spatial locations
        gram = gram / tf.cast(h*w, tf.float32)
        return gram

    def generate_features(self):
        """extracts the features used to calculate neural style cost"""
        # Load the VGG19 model for preprocessing
        VGG19 = tf.keras.applications.vgg19
        # Preprocessing fucntion expect the values in range [0, 255]
        style_image = VGG19.preprocess_input(self.style_image * 255)
        # Our architecutre only uses the last layer for content
        style_features = self.model(style_image)[:-1]
        style_matrix = [self.gram_matrix(tf.convert_to_tensor(feature))
                        for feature in style_features]

        content_image = VGG19.preprocess_input(self.content_image * 255)
        content_feature = self.model(content_image)[-1]

        self.gram_style_features = style_matrix
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        "Calculates the style cost for a single layer"""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")

        if tf.rank(style_output) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        # Number of channels
        c = style_output.shape[-1]

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(f"gram_target must be a \
tensor of shape [1, {c}, {c}]")

        if gram_target.shape != (1, c, c):
            raise TypeError(f"gram_target must be a \
tensor of shape [1, {c}, {c}]")

        gram_output = self.gram_matrix(style_output)
        style_cost = tf.reduce_mean(tf.square(gram_output - gram_target))
        return style_cost

    def style_cost(self, style_outputs):
        """Calculates the total style cost"""
        length = len(style_outputs)
        len_style = len(self.style_layers)

        if not isinstance(style_outputs, list) or length != len_style:
            raise TypeError(f"style_outputs must be \
a list with a length of {len_style}")

        # Sets the weight so that they sum to one
        weight = 1 / length
        style_cost = 0
        # We get the cost of each layer and add it to the total
        for i in range(length):
            style_cost += weight * self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i])
        return style_cost

    def content_cost(self, content_output):
        """Calculates the content cost"""
        feature_shape = self.content_feature.shape

        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(f"content_output must be a tensor \
of shape {feature_shape}")

        if content_output.shape != feature_shape:
            raise TypeError(f"content_output must be a tensor \
of shape {feature_shape}")

        content_target = self.content_feature
        content_cost = tf.reduce_mean(
                       tf.square(content_output - content_target))
        return content_cost

    def total_cost(self, generated_image):
        """
            method calculate total cost for the generated image

        :param generated_image: tf.Tensor, shape(1,nh,nw,3) generated image

        :return: (J, J_content, J_style)
                J: total cost
                J_content: content cost
                J_style: style cost
        """
        shape_content_image = self.content_image.shape

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != shape_content_image):
            raise TypeError("generated_image must be a tensor of shape {}"
                            .format(shape_content_image))

        # preprocess generated img
        preprocess_generated_image = \
            (tf.keras.applications.
             vgg19.preprocess_input(generated_image * 255))

        # calculate content and style for generated image
        generated_output = self.model(preprocess_generated_image)

        # def content and style
        generated_content = generated_output[-1]
        generated_style = generated_output[:-1]

        J_content = self.content_cost(generated_content)
        J_style = self.style_cost(generated_style)
        J_var = self.variational_cost(generated_image)
        J = self.alpha * J_content + self.beta * J_style + self.var * J_var

        return J, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Computes the gradients of the generated image"""
        shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(shape))

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            total_cost, content_cost, \
                style_cost, var_cost = self.total_cost(generated_image)
        grads = tape.gradient(total_cost, generated_image)
        return grads, total_cost, content_cost, style_cost, var_cost

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9,
                       beta2=0.99):
        """ Method to generate the neural style transfered image """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step is not None and (step <= 0 or step >= iterations):
            raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        # intialize image
        generated_image = tf.Variable(self.content_image)

        # intialize best cost and best image
        best_cost = float('inf')
        best_image = None

        # Initialize Adam
        optimizer = tf.optimizers.Adam(lr, beta1, beta2)

        # Optimization loop
        for i in range(iterations + 1):
            # compute gradients and costs
            grads, J_total, J_content, J_style, J_var = (
                    self.compute_grads(generated_image))

            # use opt
            optimizer.apply_gradients([(grads, generated_image)])

            # selected best cost and best image
            if J_total < best_cost:
                best_cost = float(J_total)
                best_image = generated_image

            # Print step required
            if step is not None and (i % step == 0 or i == iterations):
                print("Cost at iteration {}: {}, content {}, style {}, var {}"
                      .format(i, J_total, J_content, J_style, J_var))

        # remove sup dim
        best_image = best_image[0]
        best_image = tf.clip_by_value(best_image, 0, 1)
        best_image = best_image.numpy()

        return best_image, best_cost

    @staticmethod
    def variational_cost(generated_image):
        """
        calculate total cost for the generated image
        """
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or (len(generated_image.shape) != 4
                    and len(generated_image.shape) != 3)):
            raise TypeError('image must be a tensor of rank 3 or 4')

        variational_loss = tf.image.total_variation(generated_image)
        # Remove the extra dimension
        variational_loss = tf.squeeze(variational_loss)

        return variational_loss

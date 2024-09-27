#!/usr/bin/env python3
"""
Exercice 4 : LeNet-5 in Tensorflow
"""
# Description of the purpose of the code
import tensorflow as tf  # Importing the tensorflow module for deep learning


def lenet5(x, y):
    """function that builds a modified version of LeNet-5"""
    # Function definition with input parameters and docstring

    # Initializer for the layer weights
    initializer = tf.contrib.layers.variance_scaling_initializer()
    layer = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        kernel_initializer=initializer,
        activation=tf.nn.relu)  # Convolutional layer with 6 filters,
    # kernel size 5x5, same padding, ReLU activation
    output = layer(x)  # Apply the convolutional layer to the input
    layer = tf.layers.MaxPooling2D(pool_size=2,
                                   strides=2)  # Max pooling layer
    # with pool size 2x2 and stride 2
    output = layer(output)  # Apply the max pooling layer to the output
    layer = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding='valid',
                             kernel_initializer=initializer,
                             # Convolutional layer with 16 filters,
                             # kernel size 5x5, valid padding, ReLU activation
                             activation=tf.nn.relu)
    output = layer(output)  # Apply the convolutional layer to the output
    layer = tf.layers.MaxPooling2D(pool_size=2,
                                   # Max pooling layer with
                                   # pool size 2x2 and stride 2
                                   strides=2)
    output = layer(output)  # Apply the max pooling layer to the output
    layer = tf.layers.Flatten()  # Flatten the output to a 1D tensor
    output = layer(output)  # Apply the flatten layer to the output
    layer = tf.layers.Dense(units=120,
                            activation=tf.nn.relu,
                            # Fully connected layer with 120 units,
                            # ReLU activation
                            kernel_initializer=initializer)
    output = layer(output)  # Apply the fully connected layer to the output
    layer = tf.layers.Dense(units=84,
                            activation=tf.nn.relu,
                            # Fully connected layer with 84 units,
                            # ReLU activation
                            kernel_initializer=initializer)
    output = layer(output)  # Apply the fully connected layer to the output
    layer = tf.layers.Dense(units=10,
                            # Fully connected layer with 10 units
                            kernel_initializer=initializer)
    output = layer(output)  # Apply the fully connected layer to the output

    # Define loss as softmax cross-entropy between the
    # true labels and unactivated output (logits)
    loss = tf.losses.softmax_cross_entropy(y, output)

    # Define an Adam optimizer with default learning rate & minimize the loss
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Activate the output with softmax
    y_pred = tf.nn.softmax(output)

    # Evaluate the accuracy of the model
    acc = accuracy(y, y_pred)

    return y_pred, train_op, loss, acc


def accuracy(y, y_pred):
    """evaluate the accuracy of the model"""
    label = tf.argmax(y, axis=1)  # Get the index of the true label
    pred = tf.argmax(y_pred, axis=1)  # Get the index of the predicted label
    # Compute the accuracy as the mean of correct predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(label, pred), tf.float32))
    return acc

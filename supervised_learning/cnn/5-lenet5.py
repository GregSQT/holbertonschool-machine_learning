#!/usr/bin/env python3
"""
Exercice 5 : LeNet-5 in Keras
"""
# Description of the purpose of the code
# Importing the tensorflow.keras module for deep learning
import tensorflow.keras as K


def lenet5(X):
    """function that builds a modified version of LeNet-5"""
    # Function definition with input parameter and docstring

    # Initializer for the layer weights
    initializer = K.initializers.he_normal()
    layer = K.layers.Conv2D(filters=6,
                            kernel_size=5,
                            padding='same',
                            kernel_initializer=initializer,
                            # Convolutional layer with 6 filters,
                            # kernel size 5x5, same padding, ReLU activation
                            activation='relu')

    output = layer(X)  # Apply the convolutional layer to the input
    layer = K.layers.MaxPool2D(pool_size=2,
                               strides=2)  # Max pooling layer
    # with pool size 2x2 and stride 2
    output = layer(output)  # Apply the max pooling layer to the output
    layer = K.layers.Conv2D(filters=16,
                            kernel_size=5,
                            padding='valid',
                            kernel_initializer=initializer,
                            # Convolutional layer with 16 filters,
                            # kernel size 5x5, valid padding, ReLU activation
                            activation='relu')
    output = layer(output)  # Apply the convolutional layer to the output
    layer = K.layers.MaxPool2D(pool_size=2,
                               # Max pooling layer with
                               # pool size 2x2 and stride 2
                               strides=2)
    output = layer(output)  # Apply the max pooling layer to the output
    layer = K.layers.Flatten()  # Flatten the output to a 1D tensor
    output = layer(output)  # Apply the flatten layer to the output
    layer = K.layers.Dense(units=120,
                           activation='relu',
                           # Fully connected layer with 120 units,
                           # ReLU activation
                           kernel_initializer=initializer)
    output = layer(output)  # Apply the fully connected layer to the output
    layer = K.layers.Dense(units=84,
                           activation='relu',
                           # Fully connected layer with 84 units,
                           # ReLU activation
                           kernel_initializer=initializer)
    output = layer(output)  # Apply the fully connected layer to the output
    # Here, pass 'softmax' activation to the
    # model prior to compiling/training the model
    # Note: It is not recommended to use softmax
    # activation as the last layer for model compilation
    layer = K.layers.Dense(units=10,
                           activation='softmax',
                           # Fully connected layer with 10 units and
                           # softmax activation
                           kernel_initializer=initializer)
    output = layer(output)  # Apply the fully connected layer to the output

    # Instantiate a model from the Model class, with inputs and outputs
    model = K.models.Model(inputs=X, outputs=output)

    # Compile the model with Adam optimizer,
    # categorical cross-entropy loss, and accuracy metric
    # 'from_logits=True' is more numerically stable,
    # assuming that y_pred does not encode a probability distribution
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

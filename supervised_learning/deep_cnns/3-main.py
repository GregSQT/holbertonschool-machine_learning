#!/usr/bin/env python3

# Import the Keras library as K
import tensorflow.keras as K

# Import the projection_block function from the '3-projection_block' module
projection_block = __import__('3-projection_block').projection_block

# Check if this script is being run as the main module
if __name__ == '__main__':
    # Define the input tensor with shape (224, 224, 3)
    X = K.Input(shape=(224, 224, 3))

    # Create the projection block using the function projection_block and passing the input tensor and filter sizes [64, 64, 256]
    Y = projection_block(X, [64, 64, 256])

    # Instantiate a model from the Model class with input X and output Y
    model = K.models.Model(inputs=X, outputs=Y)

    # Display a summary of the model's architecture, including the layer names, output shapes, and number of parameters
    model.summary()
    
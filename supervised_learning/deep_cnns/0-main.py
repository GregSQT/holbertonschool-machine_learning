#!/usr/bin/env python3
import tensorflow.keras as K
# Import the inception_block function from the '0-inception_block' module
inception_block = __import__('0-inception_block').inception_block


if __name__ == '__main__':
    # Define the input tensor with shape (224, 224, 3)
    X = K.Input(shape=(224, 224, 3))
    
    # Call the inception_block function with the input tensor and filters list [64, 96, 128, 16, 32, 32]
    # This creates an inception block with specified filters for different branches
    Y = inception_block(X, [64, 96, 128, 16, 32, 32])
    
    # Create a Keras Model with the input X and output Y
    model = K.models.Model(inputs=X, outputs=Y)
    
    # Print the summary of the model, showing the layers and their output shapes
    model.summary()
    
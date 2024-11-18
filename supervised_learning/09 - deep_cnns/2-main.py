#!/usr/bin/env python3
import tensorflow.keras as K


# Import the identity_block function from the '2-identity_block' module
identity_block = __import__('2-identity_block').identity_block

# Check if the code is being executed as the main module
if __name__ == '__main__':
    # Define the input tensor with shape (224, 224, 256)
    X = K.Input(shape=(224, 224, 256))

    # Use the identity_block function to create an identity block with the given filters [64, 64, 256]
    Y = identity_block(X, [64, 64, 256])

    # Instantiate a model from the Model class with input X and output Y
    model = K.models.Model(inputs=X, outputs=Y)

    # Print the summary of the model, showing the layers and their output shapes
    model.summary()
    
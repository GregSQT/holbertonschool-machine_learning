#!/usr/bin/env python3

# Import the Keras library as K
import tensorflow.keras as K

# Import the dense_block function from the '5-dense_block' module
dense_block = __import__('5-dense_block').dense_block

if __name__ == '__main__':
    # Define the input tensor with shape (56, 56, 64)
    X = K.Input(shape=(56, 56, 64))

    # Call the dense_block function with input X, 64 filters, growth rate of 32, and 6 layers
    # The function returns two values: Y and nb_filters
    Y, nb_filters = dense_block(X, 64, 32, 6)

    # Instantiate a model from the Model class with input X and output Y
    model = K.models.Model(inputs=X, outputs=Y)

    # Print the model summary showing the architecture of the model
    model.summary()

    # Print the value of nb_filters, which indicates the number of filters in the final output
    print(nb_filters)
    
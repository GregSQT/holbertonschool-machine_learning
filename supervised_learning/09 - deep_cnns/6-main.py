#!/usr/bin/env python3

# Import the Keras library as K
import tensorflow.keras as K

# Import the transition_layer function from the '6-transition_layer' module
transition_layer = __import__('6-transition_layer').transition_layer

if __name__ == '__main__':
    # Define the input tensor with shape (56, 56, 256)
    X = K.Input(shape=(56, 56, 256))

    # Apply the transition_layer function to the input tensor with 256 filters and compression factor 0.5
    Y, nb_filters = transition_layer(X, 256, 0.5)

    # Instantiate a model from the Model class with input X and output Y
    model = K.models.Model(inputs=X, outputs=Y)

    # Print the summary of the model
    model.summary()

    # Print the number of filters in the output after the transition layer
    print(nb_filters)
    
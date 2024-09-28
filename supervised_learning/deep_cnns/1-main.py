#!/usr/bin/env python3
import tensorflow.keras as K


# Import the 'inception_network' function from the '1-inception_network' module
inception_network = __import__('1-inception_network').inception_network

if __name__ == '__main__':
    # Create the Inception Network model using the 'inception_network' function
    model = inception_network()

    # Print a summary of the model architecture, including layer names and output shapes
    model.summary()
    
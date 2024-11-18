#!/usr/bin/env python3

# Import the Keras library as K
import tensorflow.keras as K

# Import the resnet50 function from the '4-resnet50' module
resnet50 = __import__('4-resnet50').resnet50

# Check if this script is the main module being run
if __name__ == '__main__':
    # Call the resnet50 function to build the ResNet-50 model
    model = resnet50()

    # Display a summary of the model architecture
    model.summary()
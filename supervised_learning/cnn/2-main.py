#!/usr/bin/env python3
import numpy as np  # Importing the numpy module for numerical computations

conv_backward = __import__('2-conv_backward').conv_backward  # Importing the conv_backward function from the specified module

if __name__ == "__main__":
    # Execute the following code only if the script is run directly and not imported as a module

    np.random.seed(0)  # Setting the random seed for reproducibility

    # Load the MNIST dataset from the specified file
    lib = np.load('data/MNIST.npz')
    X_train = lib['X_train']  # Get the training images from the loaded dataset
    _, h, w = X_train.shape  # Get the height and width of the images
    X_train_c = X_train[:10].reshape((-1, h, w, 1))  # Select a subset of training images and reshape them

    W = np.random.randn(3, 3, 1, 2)  # Randomly initialize the filter weights
    b = np.random.randn(1, 1, 1, 2)  # Randomly initialize the biases

    dZ = np.random.randn(10, h - 2, w - 2, 2)  # Randomly initialize the gradient of the output

    print(conv_backward(dZ, X_train_c, W, b, padding="valid"))  # Perform backpropagation on the convolutional layer
    
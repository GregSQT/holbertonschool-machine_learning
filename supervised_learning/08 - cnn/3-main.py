#!/usr/bin/env python3
import numpy as np  # Importing the numpy module for numerical computations

pool_backward = __import__('3-pool_backward').pool_backward  # Importing the pool_backward function from the specified module

if __name__ == "__main__":
    # Execute the following code only if the script is run directly and not imported as a module

    np.random.seed(0)  # Setting the random seed for reproducibility

    # Load the MNIST dataset from the specified file
    lib = np.load('data/MNIST.npz')
    X_train = lib['X_train']  # Get the training images from the loaded dataset
    _, h, w = X_train.shape  # Get the height and width of the images
    X_train_a = X_train[:10].reshape((-1, h, w, 1))  # Select a subset of training images and reshape them
    X_train_b = 1 - X_train_a  # Invert the training images to create a second channel
    X_train_c = np.concatenate((X_train_a, X_train_b), axis=3)  # Concatenate the two channels to create a new array

    dA = np.random.randn(10, h // 3, w // 3, 2)  # Randomly initialize the gradient of the output

    print(pool_backward(dA, X_train_c, (3, 3), stride=(3, 3)))  # Perform backpropagation on the pooling layer
    
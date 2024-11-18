#!/usr/bin/env python3
import matplotlib.pyplot as plt  # Importing the matplotlib.pyplot module for plotting
import numpy as np  # Importing the numpy module for numerical computations


conv_forward = __import__('0-conv_forward').conv_forward  # Importing the conv_forward function from the specified module

if __name__ == "__main__":
    # Execute the following code only if the script is run directly and not imported as a module
    
    np.random.seed(0)  # Setting the random seed for reproducibility
    
    # Load the MNIST dataset from the specified file
    lib = np.load('data/MNIST.npz')
    X_train = lib['X_train']  # Get the training images from the loaded dataset
    m, h, w = X_train.shape  # Get the number of training examples, height, and width of the images
    X_train_c = X_train.reshape((-1, h, w, 1))  # Reshape the training images to have a single channel

    # Initialize the weights and biases for the convolutional layer
    W = np.random.randn(3, 3, 1, 2)  # Randomly initialize the filter weights
    b = np.random.randn(1, 1, 1, 2)  # Randomly initialize the biases

    def relu(Z):
        return np.maximum(Z, 0)  # ReLU activation function to apply element-wise on an input array

    # Display the first training image using matplotlib
    plt.imshow(X_train[0])
    plt.show()
    
#!/usr/bin/env python3
# Shebang line indicating the interpreter to be used when executing the script
import matplotlib.pyplot as plt  # Importing the matplotlib.pyplot module for plotting
import numpy as np  # Importing the numpy module for numerical computations


pool_forward = __import__('1-pool_forward').pool_forward  # Importing the pool_forward function from the specified module

if __name__ == "__main__":
    # Execute the following code only if the script is run directly and not imported as a module
    
    np.random.seed(0)  # Setting the random seed for reproducibility
    
    # Load the MNIST dataset from the specified file
    lib = np.load('data/MNIST.npz')
    X_train = lib['X_train']  # Get the training images from the loaded dataset
    m, h, w = X_train.shape  # Get the number of training examples, height, and width of the images
    X_train_a = X_train.reshape((-1, h, w, 1))  # Reshape the training images to have a single channel
    X_train_b = 1 - X_train_a  # Invert the training images to create a second channel
    X_train_c = np.concatenate((X_train_a, X_train_b), axis=3)  # Concatenate the two channels to create a new array

    print(X_train_c.shape)  # Print the shape of the new array
    plt.imshow(X_train_c[0, :, :, 0])  # Display the first channel of the new array using matplotlib
    plt.show()
    plt.imshow(X_train_c[0, :, :, 1])  # Display the second channel of the new array using matplotlib
    plt.show()
    A = pool_forward(X_train_c, (2, 2), stride=(2, 2))  # Apply pooling operation to the new array
    print(A.shape)  # Print the shape of the pooled array
    plt.imshow(A[0, :, :, 0])  # Display the first channel of the pooled array using matplotlib
    plt.show()
    plt.imshow(A[0, :, :, 1])  # Display the second channel of the pooled array using matplotlib
    plt.show()
    
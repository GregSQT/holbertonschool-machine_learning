#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid


if __name__ == '__main__':
    # Load the MNIST dataset
    dataset = np.load('data/MNIST.npz')
    images = dataset['X_train']  # Extract the training images from the dataset
    print(images.shape)  # Print the shape of the training images array

    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])  # Define a kernel for convolution
    images_conv = convolve_grayscale_valid(images, kernel)  # Perform convolution on the images using the kernel
    print(images_conv.shape)  # Print the shape of the convolved images array

    # Display the original image
    plt.imshow(images[0], cmap='gray')
    plt.show()

    # Display the convolved image
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
    
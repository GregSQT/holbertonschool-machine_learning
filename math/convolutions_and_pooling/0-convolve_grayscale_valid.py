#!/usr/bin/env python3
"""
Exercice 0 : Valid Convolution
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """function that performs a valid convolution on grayscale images"""
    m = images.shape[0]  # Number of images
    h = images.shape[1]  # Height of images
    w = images.shape[2]  # Width of images
    kh = kernel.shape[0]  # Height of kernel
    kw = kernel.shape[1]  # Width of kernel
    image_num = np.arange(m)  # Array of image indices
    # Output array for convolved images
    output = np.zeros(shape=(m, h - kh + 1, w - kw + 1))

    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            # Perform element-wise multiplication of the image patch & kernel
            # then sum the results along
            # the height and width dimensions (axis 1 and axis 2)
            output[image_num, i, j] = np.sum(
                images[
                    image_num,
                    i: i + kh,
                    j: j + kw
                ] * kernel,
                axis=(1, 2)
            )
    return output

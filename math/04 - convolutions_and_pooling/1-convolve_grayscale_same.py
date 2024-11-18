#!/usr/bin/env python3
"""
Exercice 1 : Same Convolution
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """function that performs a same convolution on grayscale images"""
    m = images.shape[0]  # Number of images
    h = images.shape[1]  # Height of images
    w = images.shape[2]  # Width of images
    kh = kernel.shape[0]  # Height of kernel
    kw = kernel.shape[1]  # Width of kernel
    image_num = np.arange(m)  # Array of image indices
    output = np.zeros(shape=(m, h, w))  # Output array for convolved images

    # Pad images before convolution
    # Handle even vs. odd filter sizes with np.ceil()
    ph = int(np.ceil((kh - 1) / 2))
    pw = int(np.ceil((kw - 1) / 2))

    # Pad images accordingly, padding always symmetric here
    padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    for i in range(h):
        for j in range(w):
            # Perform element-wise multiplication
            # of the padded image patch and kernel,
            # then sum the results along
            # the height and width dimensions (axis 1 and axis 2)
            output[
                image_num,
                i,
                j
            ] = np.sum(
                padded_images[
                    image_num,
                    i: i + kh,
                    j: j + kw
                ] * kernel,
                axis=(1, 2)
            )
    return output

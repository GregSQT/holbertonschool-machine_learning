#!/usr/bin/env python3
"""
Exercice 2 : Convolution with Padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """function that performs a convolution with custom padding"""
    m = images.shape[0]  # Number of images
    h = images.shape[1]  # Height of images
    w = images.shape[2]  # Width of images
    kh = kernel.shape[0]  # Height of kernel
    kw = kernel.shape[1]  # Width of kernel
    image_num = np.arange(m)  # Array of image indices
    ph = padding[0]  # Padding height
    pw = padding[1]  # Padding width

    # Pad images before convolution, padding always symmetric here
    padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Output size depends on filter size and padding
    output = np.zeros(shape=(m, h - kh + 1 + 2 * ph, w - kw + 1 + 2 * pw))

    for i in range(h - kh + 1 + 2 * ph):
        for j in range(w - kw + 1 + 2 * pw):
            # Perform element-wise multiplication of
            # the padded image patch and kernel,
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

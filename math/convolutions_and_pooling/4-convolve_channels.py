#!/usr/bin/env python3
"""
Exercice 4 : Convolution with Channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """function that performs a convolution on images with channels"""
    m = images.shape[0]  # Number of images
    h = images.shape[1]  # Height of images
    w = images.shape[2]  # Width of images
    c = images.shape[3]  # Number of channels
    kh = kernel.shape[0]  # Height of kernel
    kw = kernel.shape[1]  # Width of kernel
    image_num = np.arange(m)  # Array of image indices
    sh = stride[0]  # Stride height
    sw = stride[1]  # Stride width

    if isinstance(padding, tuple):
        ph = padding[0]  # Padding height
        pw = padding[1]  # Padding width
    elif padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        # Output size depends on filter size and must be equal to image size
        # Imposing constraints on padding for a given set of strides
        ph = int(np.ceil(((sh * h) - sh + kh - h) / 2))
        pw = int(np.ceil(((sw * w) - sw + kw - w) / 2))

    if isinstance(padding, tuple) or padding == 'same':
        # Pad images before convolution, padding always symmetric here
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')

    output = np.zeros(shape=(m,
                             int((h - kh + 2 * ph) / sh + 1),
                             int((w - kw + 2 * pw) / sw + 1)))

    for i in range(int((h - kh + 2 * ph) / sh + 1)):
        for j in range(int((w - kw + 2 * pw) / sw + 1)):
            # Perform element-wise multiplication of
            # the strided image patch and kernel,
            # then sum the results along the height,
            # width, and channel dimensions (axis 1, 2, 3)
            output[
                image_num,
                i,
                j
            ] = np.sum(
                images[
                    image_num,
                    i * sh: i * sh + kh,
                    j * sw: j * sw + kw
                ] * kernel,
                axis=(1, 2, 3)
            )
    return output

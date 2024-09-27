#!/usr/bin/env python3
""""
Exercice 6 : Pooling
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """function that performs pooling on images"""
    m = images.shape[0]  # Number of images
    h = images.shape[1]  # Height of images
    w = images.shape[2]  # Width of images
    c = images.shape[3]  # Number of channels
    kh = kernel_shape[0]  # Height of pooling kernel
    kw = kernel_shape[1]  # Width of pooling kernel
    image_num = np.arange(m)  # Array of image indices
    sh = stride[0]  # Stride height
    sw = stride[1]  # Stride width
    func = {'max': np.max, 'avg': np.mean}  # Pooling function dictionary

    output = np.zeros(shape=(m,
                             int((h - kh) / sh + 1),
                             int((w - kw) / sw + 1),
                             c))  # Output array to store pooled images

    if mode in ['max', 'avg']:
        # Perform pooling based on the specified mode
        for i in range(int((h - kh) / sh + 1)):
            for j in range(int((w - kw) / sw + 1)):
                # Extract the pooling region from the images
                pooling_region = images[
                    image_num,
                    i * sh: i * sh + kh,
                    j * sw: j * sw + kw,
                    :
                ]
                # Apply the pooling function to
                # the pooling region along the height and width dimensions
                pooled_value = func[mode](pooling_region, axis=(1, 2))
                # Store the pooled value in the output array
                output[
                    image_num,
                    i,
                    j,
                    :
                ] = pooled_value

    return output

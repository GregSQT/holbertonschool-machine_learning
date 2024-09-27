#!/usr/bin/env python3
"""
Exercice 0 : Forward Convolution
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """function that performs a forward propagation over convolution layer"""
    m = A_prev.shape[0]  # Number of examples in the input
    h_prev = A_prev.shape[1]  # Height of the input
    w_prev = A_prev.shape[2]  # Width of the input
    c_prev = A_prev.shape[3]  # Number of channels in the input
    kh = W.shape[0]  # Height of the filter
    kw = W.shape[1]  # Width of the filter
    c_prev = W.shape[2]  # Number of channels in the filter
    c_new = W.shape[3]  # Number of filters in the convolution layer
    image_num = np.arange(m)  # Array of image indices
    sh = stride[0]  # Stride value for height
    sw = stride[1]  # Stride value for width

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        # Calculate padding size for 'same' padding; match input & output size
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))

    if padding == 'same':
        # Pad the input with zeros before convolution
        A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph),
                                           (pw, pw), (0, 0)),
                        mode='constant')

    # Initialize the output with zeros
    output = np.zeros(shape=(m,
                             int((h_prev - kh + 2 * ph) / sh + 1),
                             int((w_prev - kw + 2 * pw) / sw + 1),
                             c_new))

    # Perform the convolution operation
    for k in range(c_new):
        for i in range(int((h_prev - kh + 2 * ph) / sh + 1)):
            for j in range(int((w_prev - kw + 2 * pw) / sw + 1)):
                # Apply the convolution operation at each position
                output[
                    image_num,
                    i,
                    j,
                    k
                ] = np.sum(
                    A_prev[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw
                    ] * W[:, :, :, k],
                    axis=(1, 2, 3)
                ) + b[0, 0, 0, k]

    # Apply the activation function to the output
    output = activation(output)
    return output

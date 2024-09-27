#!/usr/bin/env python3
"""
Exercice 2 : Backpropagation over Convolution Layer
"""
# Description of the purpose of the code
import numpy as np  # Importing the numpy module for numerical computations


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """function that performs a backpropagation over a convolutional layer"""
    # Function definition with input parameters and docstring

    m = dZ.shape[0]  # Number of examples in the output
    h_new = dZ.shape[1]  # Height of the output
    w_new = dZ.shape[2]  # Width of the output
    c_new = dZ.shape[3]  # Number of channels in the output
    h_prev = A_prev.shape[1]  # Height of the input
    w_prev = A_prev.shape[2]  # Width of the input
    c_prev = A_prev.shape[3]  # Number of channels in the input
    kh = W.shape[0]  # Height of the filter
    kw = W.shape[1]  # Width of the filter
    sh = stride[0]  # Stride value for height
    sw = stride[1]  # Stride value for width

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        # Calculate padding size for 'same' padding to
        # match input and output size
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))

    if padding == 'same':
        # Pad the input before convolution, padding always symmetric here
        A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw),
                                           (0, 0)),
                        mode='constant')

    dA_prev = np.zeros(shape=A_prev.shape)  # Initialize gradient of
    # the previous layer (input)
    dW = np.zeros(shape=W.shape)  # Initialize gradient of the weights
    db = np.zeros(shape=b.shape)  # Initialize gradient of the biases

    for img_num in range(m):
        for k in range(c_new):
            for i in range(h_new):
                for j in range(w_new):
                    # Compute gradients of the previous layer, weights,
                    # and biases using the chain rule

                    dA_prev[
                        img_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ] += dZ[
                        img_num,
                        i,
                        j,
                        k
                    ] * W[:, :, :, k]

                    dW[:, :, :, k] += A_prev[
                        img_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ] * dZ[
                        img_num,
                        i,
                        j,
                        k
                    ]

    # Compute the gradient of the biases
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        dA_prev = dA_prev[
            :,
            ph: dA_prev.shape[1] - ph,
            pw: dA_prev.shape[2] - pw,
            :
        ]  # Remove padding from the gradient of the previous layer

    return dA_prev, dW, db  # Return the computed gradients

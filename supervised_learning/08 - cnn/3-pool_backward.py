#!/usr/bin/env python3
"""
Exercice 3 : Backpropagation over Pooling Layer
"""
# Description of the purpose of the code
import numpy as np  # Importing the numpy module for numerical computations


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs a backpropagation over a pooling layer"""
    # Function definition with input parameters and docstring

    m = dA.shape[0]  # Number of examples in the output
    h_new = dA.shape[1]  # Height of the output
    w_new = dA.shape[2]  # Width of the output
    c = dA.shape[3]  # Number of channels in the output
    h_prev = A_prev.shape[1]  # Height of the input
    w_prev = A_prev.shape[2]  # Width of the input
    kh = kernel_shape[0]  # Height of the pooling kernel
    kw = kernel_shape[1]  # Width of the pooling kernel
    sh = stride[0]  # Stride value for height
    sw = stride[1]  # Stride value for width
    # Dictionary mapping pooling mode to the corresponding numpy function
    func = {'max': np.max, 'avg': np.mean}

    # Initialize gradient of the previous layer (input)
    dA_prev = np.zeros(shape=A_prev.shape)

    if mode in ['max', 'avg']:
        # Perform backpropagation based on the specified mode

        for img_num in range(m):
            for k in range(c):
                for i in range(h_new):
                    for j in range(w_new):
                        window = A_prev[
                            img_num,
                            i * sh: i * sh + kh,
                            j * sw: j * sw + kw,
                            k
                        ]  # Get the corresponding window from the input

                        if mode == 'max':
                            # For max pooling, the derivative
                            # is 1 for the max value and 0 for others
                            # Create a mask of 1 and 0s where
                            # 1 corresponds to the max value
                            mask = np.where(window == np.max(window), 1, 0)

                        elif mode == 'avg':
                            # For average pooling, the derivative is
                            # equal to the reciprocal of the window size
                            # Create a mask of ones divided by the window size
                            mask = np.ones(shape=window.shape) / (kh * kw)

                        dA_prev[
                            img_num,
                            i * sh: i * sh + kh,
                            j * sw: j * sw + kw,
                            k
                        ] += mask * dA[
                            img_num,
                            i,
                            j,
                            k
                            # Propagate the gradients to
                            # the previous layer using the mask
                        ]

    return dA_prev  # Return the computed gradients of the previous layer

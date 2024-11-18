#!/usr/bin/env python3
"""
Exercice 1 : Forward Pooling
"""
# Description of the purpose of the code
import numpy as np  # Importing the numpy module for numerical computations


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs forward propagation over a pooling layer"""
    # Function definition with input parameters and docstring

    m = A_prev.shape[0]  # Number of examples in the input
    h_prev = A_prev.shape[1]  # Height of the input
    w_prev = A_prev.shape[2]  # Width of the input
    c_prev = A_prev.shape[3]  # Number of channels in the input
    kh = kernel_shape[0]  # Height of the pooling kernel
    kw = kernel_shape[1]  # Width of the pooling kernel
    image_num = np.arange(m)  # Array of image indices
    sh = stride[0]  # Stride value for height
    sw = stride[1]  # Stride value for width
    # Dictionary mapping pooling mode to the corresponding numpy function
    func = {'max': np.max, 'avg': np.mean}

    output = np.zeros(shape=(m,
                             int((h_prev - kh) / sh + 1),
                             int((w_prev - kw) / sw + 1),
                             c_prev))  # Initialize the output with zeros

    if mode in ['max', 'avg']:
        # Perform pooling operation based on the specified mode

        for i in range(int((h_prev - kh) / sh + 1)):
            for j in range(int((w_prev - kw) / sw + 1)):
                # Iterate through the input tensor and apply pooling operation

                output[
                    image_num,
                    i,
                    j,
                    :
                ] = func[mode](
                    A_prev[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ], axis=(1, 2)
                )

    return output

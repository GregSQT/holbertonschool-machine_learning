#!/usr/bin/env python3
"""
DenseNet-121
"""

import tensorflow.keras as K

# Import the dense_block & transition_layer functions from respective modules
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    function that builds a DenseNet-121 network
    as described in Densely Connected Convolutional Networks
    """
    # Use the He normal initializer for the weights
    initializer = K.initializers.he_normal()

    # Define the input tensor with shape (224, 224, 3)
    X = K.Input(shape=(224, 224, 3))

    # Apply BatchNormalization to the input tensor
    norm_1 = K.layers.BatchNormalization()
    output_1 = norm_1(X)

    # Apply ReLU activation to the normalized input
    activ_1 = K.layers.Activation('relu')
    output_1 = activ_1(output_1)

    # Layer 1: 7x7 Convolution with 64 filters, stride 2, and no activation
    layer_1 = K.layers.Conv2D(filters=64,
                              kernel_size=7,
                              padding='same',
                              strides=2,
                              kernel_initializer=initializer,
                              activation=None)
    output_1 = layer_1(output_1)

    # Layer 2: 3x3 MaxPooling with stride 2
    layer_2 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)
    output_2 = layer_2(output_1)

    # Apply the first dense block with 6 layers and specified growth rate
    db1_output = dense_block(output_2, output_2.shape[-1], growth_rate, 6)

    # Apply the first transition layer with compression factor
    tl1_output = transition_layer(db1_output[0],
                                  int(db1_output[1]), compression)

    # Apply the second dense block with 12 layers and specified growth rate
    db2_output = dense_block(tl1_output[0], tl1_output[1], growth_rate, 12)

    # Apply the second transition layer with compression factor
    tl2_output = transition_layer(db2_output[0],
                                  int(db2_output[1]), compression)

    # Apply the third dense block with 24 layers and specified growth rate
    db3_output = dense_block(tl2_output[0], tl2_output[1], growth_rate, 24)

    # Apply the third transition layer with compression factor
    tl3_output = transition_layer(db3_output[0],
                                  int(db3_output[1]), compression)

    # Apply the fourth dense block with 16 layers and specified growth rate
    db4_output = dense_block(tl3_output[0], tl3_output[1], growth_rate, 16)

    # Layer 3: 7x7 AveragePooling to reduce data to 1x1
    layer_3 = K.layers.AvgPool2D(pool_size=7,
                                 padding='same',
                                 strides=None)
    output_3 = layer_3(db4_output[0])

    # Define a dense layer with softmax activation for final classification
    softmax = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2())
    output_4 = softmax(output_3)

    # Instantiate a model from the Model class with input X & output output_4
    model = K.models.Model(inputs=X, outputs=output_4)

    return model

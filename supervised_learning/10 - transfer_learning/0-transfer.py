#!/usr/bin/env python3
""" 
Transfer learning with CNN models on the Cifar10 dataset using ResNetV2
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-process the data for the model.

    Args:
        X (np.ndarray): Array of shape (m, 32, 32, 3) containing the CIFAR 10
            images.
        Y (np.ndarray): Array of shape (m,) containing the CIFAR 10 labels.

    Returns:
        X_p (np.ndarray): The preprocessed X tensor.
        Y_p (np.ndarray): The preprocessed Y tensor.
    """
    X = K.applications.inception_resnet_v2.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == "__main__":
    # Load CIFAR-10 data
    (X_train, y_train), (X_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Create the base model from the pre-trained InceptionResNetV2
    base_model = K.applications.InceptionResNetV2(weights='imagenet',
                                                  include_top=False,
                                                  input_shape=(299, 299, 3))

    # Create input layer resizing from 32x32 to 299x299
    input_layer = K.layers.Input(shape=(32, 32, 3))
    resized_input = K.layers.Lambda(
        lambda img: K.preprocessing.image.smart_resize(img, (299, 299))
        )(input_layer)

    # Pass the resized input through the base model
    x = base_model(resized_input, training=False)

    # Freeze layers before a certain point
    for layer in base_model.layers[:600]:
        layer.trainable = False

    for layer in base_model.layers[600:]:
        layer.trainable = True

    # Add global average pooling and dense layers
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256,
                       activation='relu',
                       kernel_regularizer=K.regularizers.l2(0.001))(x)
    x = K.layers.Dropout(0.4)(x)
    output_layer = K.layers.Dense(10, activation='softmax')(x)

    # Create the final model
    model = K.models.Model(inputs=input_layer, outputs=output_layer)

    # Display the model architecture
    model.summary()

    # Compile the model
    model.compile(optimizer=K.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        batch_size=256,
                        epochs=12,
                        verbose=1)

    # Save the trained model
    model.save("cifar10.h5")
    
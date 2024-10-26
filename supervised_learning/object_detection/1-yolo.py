#!/usr/bin/env python3
"""
 a class Yolo (Based on 0-yolo.py)
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K


class Yolo:
    """Define the YOLO class for object detection using YOLO v3 algorithm"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the YOLO object.

        Parameters:
        - model_path (str): Path to the Darknet Keras model.
        - classes_path (str): Path to the file containing class names used by
        the model.
        - class_t (float): Box score threshold for the initial filtering step.
        - nms_t (float): IOU (Intersection over Union) threshold for non-max
        suppression.
        - anchors (numpy.ndarray): Array containing anchor box dimensions.

        Attributes:
        - model (tensorflow.keras.Model): Loaded YOLO model.
        - class_names (list): List of class names used by the model.
        - class_t (float): Box score threshold.
        - nms_t (float): Non-max suppression threshold.
        - anchors (numpy.ndarray): Anchor box dimensions.
        """
        self.model = K.models.load_model(model_path)  # Load the YOLO model
        with open(classes_path, 'r') as f:
            self.class_names = [class_name[:-1]
                                for class_name in f]  # Read class names file
        self.class_t = class_t  # Set class score threshold
        self.nms_t = nms_t  # Set non-max suppression threshold
        self.anchors = anchors  # Set anchor boxes

    def sigmoid(self, array):
        """
        Calculate the sigmoid activation function.

        Parameters:
        - array (numpy.ndarray): Input array.

        Returns:
        - numpy.ndarray: Result of the sigmoid activation applied to
        the input array.
        """
        return 1 / (1 + np.exp(-1 * array))  # Sigmoid activation function

    def process_outputs(self, outputs, image_size):
        """
        Process single-image predictions from the YOLO model.

        Parameters:
        - outputs (list of numpy.ndarray): Predictions from the
        Darknet model for a single image.
        - image_size (numpy.ndarray): Original size of the image
        [image_height, image_width].

        Returns:
        - Tuple of (boxes, box_confidences, box_class_probs):
          - boxes: List of numpy.ndarrays of shape (grid_height,
          grid_width, anchor_boxes, 4)
                   containing the processed boundary boxes for each output.
          - box_confidences: List of numpy.ndarrays of shape (grid_height,
          grid_width, anchor_boxes, 1)
                             containing the box confidences for each output.
          - box_class_probs: List of numpy.ndarrays of shape (grid_height,
                            grid_width, anchor_boxes, classes)
                            containing the box’s class probabilities for
                            each output.
        """
        # Initialize lists to store processed data
        boxes = []
        box_confidences = []
        box_class_probs = []

        # Loop over the output feature maps
        for i, output in enumerate(outputs):
            grid_height = output.shape[0]
            grid_width = output.shape[1]
            anchor_boxes = output.shape[2]
            boxs = output[..., :4]
            t_x = boxs[..., 0]
            t_y = boxs[..., 1]
            t_w = boxs[..., 2]
            t_h = boxs[..., 3]

            # Create 3D arrays with the left-corner coordinates (c_x, c_y) of
            # each grid cell
            c_x = np.arange(grid_width).reshape(1, grid_width)
            c_x = np.repeat(c_x, grid_height, axis=0)
            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)
            c_y = np.arange(grid_width).reshape(1, grid_width)
            c_y = np.repeat(c_y, grid_height, axis=0).T
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)

            # Calculate bounding box coordinates
            b_x = (self.sigmoid(t_x) + c_x) / grid_width
            b_y = (self.sigmoid(t_y) + c_y) / grid_height

            # Calculate bounding box dimensions
            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]
            image_width = self.model.input_shape[1]
            image_height = self.model.input_shape[2]
            b_w = (anchor_width * np.exp(t_w)) / image_width
            b_h = (anchor_height * np.exp(t_h)) / image_height

            # Calculate box coordinates relative to the original image
            x_1 = b_x - b_w / 2
            y_1 = b_y - b_h / 2
            x_2 = x_1 + b_w
            y_2 = y_1 + b_h

            # Express the boundary box coordinates relative to the original
            # image
            x_1 *= image_size[1]
            y_1 *= image_size[0]
            x_2 *= image_size[1]
            y_2 *= image_size[0]

            # Update boxes according to the bounding box coordinates
            boxs[..., 0] = x_1
            boxs[..., 1] = y_1
            boxs[..., 2] = x_2
            boxs[..., 3] = y_2

            # Append the boxes coordinates to the boxes list
            boxes.append(boxs)

            # Extract the network output box_confidence prediction
            box_confidence = output[..., 4:5]
            # The prediction is passed through a sigmoid function,
            # which squashes the output in a range from 0 to 1,
            # to be interpreted as a probability.
            box_confidence = self.sigmoid(box_confidence)

            # Append box_confidence to box_confidences
            box_confidences.append(box_confidence)

            # Extract the network ouput class_probability predictions
            classes = output[..., 5:]
            # The predictions are passed through a sigmoid function,
            # which squashes the output in a range from 0 to 1,
            # to be interpreted as a probability.
            classes = self.sigmoid(classes)

            # Append class_probability predictions to box_class_probs
            box_class_probs.append(classes)

        return boxes, box_confidences, box_class_probs

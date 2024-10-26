#!/usr/bin/env python3
"""
Implementation of the Yolo class for object detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
     a class Yolo that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the Yolo object detector.

        Args:
        - model_path (str): Path to the Darknet Keras model.
        - classes_path (str): Path to the list of class names.
        - class_t (float): Box score threshold for initial filtering.
        - nms_t (float): IOU threshold for non-max suppression.
        - anchors (numpy.ndarray): Array of anchor boxes.
        """
        # Load the Darknet Keras model
        self.model = K.models.load_model(model_path)
        # Load the list of class names
        self.class_names = self.load_class_names(classes_path)
        # Set the box score threshold for initial filtering
        self.class_t = class_t
        # Set the IOU threshold for non-max suppression
        self.nms_t = nms_t
        # Set the array of anchor boxes
        self.anchors = anchors

    def load_class_names(self, file_path):
        """
        Load class names from a file.

        Args:
        - file_path (str): Path to the file containing class names.

        Returns:
        - list: List of class names.
        """
        # Open the file and read class names into a list
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]

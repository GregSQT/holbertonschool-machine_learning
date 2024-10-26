#!/usr/bin/env python3
"""
A class Yolo (Based on 4-yolo.py)
"""
import numpy as np
import tensorflow.keras as K
import os
import cv2


class Yolo:
    """
    Class that uses Yolo v3 algorithm to perform object detection

    public instance attributes:
        model: the Darknet Keras model
        class_names: list of all the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class for performing object detection
        using Yolo v3 algorithm.
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_class_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_class_names(self, classes_path):
        """
        Loads class names from the file specified in classes_path.
        """
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet model outputs to extract boundary boxes,
        box confidences,
        and class probabilities.

        Parameters:
        - outputs: list of numpy.ndarrays containing the predictions from
          the Darknet model for a single image
        - image_size: numpy.ndarray containing the image’s original size
          [image_height, image_width]

        Returns:
        A tuple of (boxes, box_confidences, box_class_probs)
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes = output.shape[:3]

            # Extract the box attributes t_x, t_y, t_w, t_h
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # Calculate the center (bx, by) and dimensions (bw, bh)
            bx = (
                self._sigmoid(tx) + np.arange(grid_width).reshape(
                    1, grid_width, 1)) / grid_width
            by = (
                self._sigmoid(ty) + np.arange(grid_height).reshape(
                    grid_height, 1, 1)) / grid_height
            bw = (
                np.exp(tw) * self.anchors[i][:, 0].reshape(
                    1, 1, anchor_boxes)) / self.model.input.shape[1]
            bh = (
                np.exp(th) * self.anchors[i][:, 1].reshape(
                    1, 1, anchor_boxes)) / self.model.input.shape[2]

            # Convert box (bx, by, bw, bh) to corner coordinates
            # (x1, y1, x2, y2)
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

            # Extract the box confidence scores (objectness)
            box_confidence = self._sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)

            # Extract the class probabilities
            class_probs = self._sigmoid(output[..., 5:])
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    @staticmethod
    def _sigmoid(x):
        """ Sigmoid activation function """
        return 1 / (1 + np.exp(-x))

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the bounding boxes based on their confidence scores
        and class probabilities.

        Args:
            boxes (list of numpy.ndarray):
            A list where each element is a numpy array
            of shape (grid_height, grid_width, anchor_boxes, 4)
            containing the bounding boxes for each output.
            box_confidences (list of numpy.ndarray):
            A list where each element is a numpy array
            of shape (grid_height, grid_width, anchor_boxes, 1)
            containing the box confidences.
            box_class_probs (list of numpy.ndarray):
            A list where each element is a numpy array
            of shape (grid_height, grid_width, anchor_boxes, classes)
            containing the class probabilities for each box.

        Returns:
            tuple:
                - filtered_boxes (numpy.ndarray): A numpy array of shape (?, 4)
                containing the filtered bounding boxes.
                - box_classes (numpy.ndarray): A numpy array of shape (?,)
                containing the predicted class index for each box.
                - box_scores (numpy.ndarray): A numpy array of shape (?)
                containing the confidence score for each box.
        """

        filtered_boxes = []
        box_classes = []
        box_scores = []

        # Iterate through each YOLO output (multiple grid sizes)
        for i in range(len(boxes)):
            # Reshape the boxes, confidences, and class probabilities
            boxes_i = boxes[i].reshape(-1, 4)
            box_confidences_i = box_confidences[i].reshape(-1)
            box_class_probs_i = box_class_probs[i].reshape(
                -1, box_class_probs[i].shape[-1])

            # Multiply confidences by class probabilities to get
            # the box scores for each class
            box_scores_i = box_confidences_i[:, np.newaxis] * box_class_probs_i

            # Get the max score and corresponding class index for each box
            # Class with the highest probability
            box_classes_i = np.argmax(box_scores_i, axis=-1)
            # Highest score across classes
            box_scores_i = np.max(box_scores_i, axis=-1)

            # Filter out boxes where the score is less than the threshold
            mask = box_scores_i >= self.class_t

            # Keep only boxes above the threshold
            filtered_boxes.append(boxes_i[mask])
            # Keep the corresponding class
            box_classes.append(box_classes_i[mask])
            # Keep the corresponding score
            box_scores.append(box_scores_i[mask])

        # Concatenate the results from all grid sizes
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Maximum Suppression (NMS) to filter out overlapping
        bounding boxes.

        Args:
            filtered_boxes (numpy.ndarray): Array of shape (?, 4) containing
                                            the filtered bounding boxes.
            box_classes (numpy.ndarray): Array of shape (?,) containing the
                                        class index for each box.
            box_scores (numpy.ndarray): Array of shape (?) containing the
                                        score for each box.

        Returns:
            tuple:
                box_predictions (numpy.ndarray): Array of shape (?, 4)
                                    containing the predicted bounding boxes.
                predicted_box_classes (numpy.ndarray): Array of shape (?,)
                                                containing the class index for
                                                    the predicted boxes.
                predicted_box_scores (numpy.ndarray): Array of shape (?)
                                                containing the score for the
                                                    predicted boxes.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Get unique classes present in the boxes
        unique_classes = np.unique(box_classes)

        # Apply NMS for each unique class
        for cls in unique_classes:
            # Get the indices for boxes of this class
            cls_mask = np.where(box_classes == cls)

            # Extract the boxes and scores for this class
            class_boxes = filtered_boxes[cls_mask]
            class_box_scores = box_scores[cls_mask]

            if len(class_boxes) == 0:
                continue  # Skip if there are no boxes for this class

            # Apply NMS using the helper method
            indices = self.nms(class_boxes, class_box_scores)

            if len(indices) > 0:  # Ensure some indices were selected by NMS
                # Append the results for this class
                box_predictions.append(class_boxes[indices])
                predicted_box_classes.append(np.full(len(indices), cls))
                predicted_box_scores.append(class_box_scores[indices])

        # Check if any boxes were kept
        if len(box_predictions) > 0:
            # Concatenate results across all classes
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0)
            predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)
        else:
            # If no boxes were found, return empty arrays
            box_predictions = np.array([])
            predicted_box_classes = np.array([])
            predicted_box_scores = np.array([])

        return box_predictions, predicted_box_classes, predicted_box_scores

    def nms(self, boxes, scores):
        """
        Non-Maximum Suppression (NMS) on boxes based on their
        Intersection Over Union (IoU).

        Args:
            boxes: numpy.ndarray of shape (?, 4) containing bounding boxes.
            scores: numpy.ndarray of shape (?) containing confidence scores.

        Returns:
            numpy.ndarray: Indices of the boxes to keep after NMS.
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Calculate IoU with the top box
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep only boxes with IoU less than the threshold
            indices_to_keep = np.where(iou <= self.nms_t)[0]
            order = order[indices_to_keep + 1]

        return np.array(keep)

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a given folder and returns them as a list of
        numpy.ndarrays along with their file paths.

        Args:
            folder_path (str): Path to the folder holding all images.

        Returns:
            tuple: (images, image_paths)
                - images: list of images as numpy.ndarrays
                - image_paths: list of paths to the individual images
        """
        images = []
        image_paths = []

        # Get all file names in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if the file is a valid image file
            if (os.path.isfile(file_path) and
                    file_name.lower().endswith(('.png', '.jpg', '.jpeg'))):
                # Read the image using OpenCV
                image = cv2.imread(file_path)

                # Append the image and its path to the lists
                images.append(image)
                image_paths.append(file_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses a list of images for input into the Darknet model.

        Args:
            images (list of numpy.ndarray): List of images to preprocess.

        Returns:
            tuple: (pimages, image_shapes)
                - pimages: numpy.ndarray of preprocessed images of shape
                           (ni, input_h, input_w, 3), where ni is the number
                of images, input_h and input_w are the model input dimensions.
                - image_shapes:ndarray of shape (ni, 2) containing the original
                                height and width of each image.
        """
        input_h = self.model.input.shape[2]  # Input height of the model
        input_w = self.model.input.shape[1]  # Input width of the model

        pimages = []
        image_shapes = []

        for image in images:
            # Store original image shape (height, width)
            original_shape = image.shape[:2]  # (image_height, image_width)
            image_shapes.append(original_shape)

            # Convert image from BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize image (input_h, input_w) using inter-cubic interpolation
            resized_image = cv2.resize(
                image, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

            # Rescale the pixel values to be between [0, 1]
            normalized_image = resized_image / 255.0

            # Append the preprocessed image
            pimages.append(normalized_image)

        # Convert the list of processed images to a numpy array
        pimages = np.array(pimages)

        # Convert image_shapes to a numpy array
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

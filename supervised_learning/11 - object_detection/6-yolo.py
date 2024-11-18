#!/usr/bin/env python3
"""
A class Yolo (Based on 5-yolo.py)
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import cv2
import glob
import os


class Yolo:
    """
    Define the YOLO class for object detection using YOLO v3 algorithm
    """

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
        # Load the YOLO model
        self.model = K.models.load_model(model_path, compile=False)
        # Read class names file
        with open(classes_path, 'r') as f:
            self.class_names = [class_name[:-1] for class_name in f]
        # Set class score threshold
        self.class_t = class_t
        # Set non-max suppression threshold
        self.nms_t = nms_t
        # Set anchor boxes
        self.anchors = anchors

    def sigmoid(self, array):
        """
        Calculate the sigmoid activation function.

        Parameters:
        - array (numpy.ndarray): Input array.

        Returns:
        - numpy.ndarray: Result of the sigmoid activation applied to
        the input array.
        """
        # Sigmoid activation function
        return 1 / (1 + np.exp(-1 * array))

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
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # BONDING BOX CENTER COORDINATES (x,y)
            c_x = np.arange(grid_width).reshape(1, grid_width)
            c_x = np.repeat(c_x, grid_height, axis=0)
            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)

            c_y = np.arange(grid_height).reshape(grid_height, 1)
            c_y = np.repeat(c_y, grid_width, axis=1)
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)

            # Extract box coordinates and dimensions
            box_coords = output[..., :4]
            t_x, t_y, t_w, t_h = box_coords[..., 0], box_coords[..., 1],
            box_coords[..., 2], box_coords[..., 3]

            # Calculate bounding box coordinates
            b_x = (self.sigmoid(t_x) + c_x) / grid_width
            b_y = (self.sigmoid(t_y) + c_y) / grid_height

            # Calculate bounding box dimensions
            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]
            image_width = self.model.input_shape[1].value
            image_height = self.model.input_shape[2].value
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
            box_coords[..., 0] = x_1
            box_coords[..., 1] = y_1
            box_coords[..., 2] = x_2
            box_coords[..., 3] = y_2

            # Append the boxes coordinates to the boxes list
            boxes.append(box_coords)

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter bounding boxes based on box confidence and class probability.

        Parameters:
        - boxes (list of numpy.ndarrays): Processed boundary boxes.
        - box_confidences (list of numpy.ndarrays): Processed box confidences.
        - box_class_probs (list of numpy.ndarrays): Processed box class
        probabilities.

        Returns:
        - Tuple of (filtered_boxes, box_classes, box_scores):
          - filtered_boxes: A numpy.ndarray of shape (?, 4) containing
          all of the filtered bounding boxes.
          - box_classes: A numpy.ndarray of shape (?,) containing the
          class number that each box in filtered_boxes predicts.
          - box_scores: A numpy.ndarray of shape (?) containing the
          box scores for each box in filtered_boxes.
        """
        obj_thresh = self.class_t  # Set box score threshold for filtering.

        # Initialize lists to store filtered data
        filtered_boxes = []  # List to store filtered bounding boxes.
        box_classes = []  # List to store predicted class numbers.
        box_scores = []  # List to store box scores.

        # Loop over each output
        for i, (box_confidence, box_class_prob, box) in enumerate(
                zip(box_confidences, box_class_probs, boxes)):
            # Compute the box scores for each output feature map
            box_scores_per_output = box_confidence * box_class_prob
            max_box_scores = np.max(box_scores_per_output, axis=3).reshape(-1)

            # Determine the object class of the boxes with the max scores
            max_box_classes = np.argmax(
                box_scores_per_output, axis=3).reshape(-1)

            # Combine all the boxes in a 2D np.ndarray
            box = box.reshape(-1, 4)

            # Create the list of indices pointing to the elements
            # to be removed using the class_t (box score threshold)
            index_list = np.where(max_box_scores < obj_thresh)

            # Delete elements by index
            max_box_scores_filtered = np.delete(max_box_scores, index_list)
            max_box_classes_filtered = np.delete(max_box_classes, index_list)
            filtered_box = np.delete(box, index_list, axis=0)

            # Append the updated arrays to the respective lists
            box_scores.append(max_box_scores_filtered)
            box_classes.append(max_box_classes_filtered)
            filtered_boxes.append(filtered_box)

        # Concatenate the lists into numpy arrays
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def iou(box1, box2):
        """
        Calculate intersection over union (IOU) for two bounding boxes.

        Parameters:
        - box1 (numpy.ndarray): Coordinates of the first box [x1, y1, x2, y2].
        - box2 (numpy.ndarray): Coordinates of the second box [x1, y1, x2, y2].

        Returns:
        - float: Intersection over union (IOU) between the two boxes.
        """
        # Calculate the maximum x-coordinate of the intersection.
        xi1 = np.maximum(box1[0], box2[0])
        # Calculate the maximum y-coordinate of the intersection.
        yi1 = np.maximum(box1[1], box2[1])
        # Calculate the minimum x-coordinate of the intersection.
        xi2 = np.minimum(box1[2], box2[2])
        # Calculate the minimum y-coordinate of the intersection.
        yi2 = np.minimum(box1[3], box2[3])
        # Calculate the area of intersection.
        inter_area = np.maximum(yi2 - yi1, 0) * np.maximum(xi2 - xi1, 0)

        # Calculate the area of the first box.
        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
        # Calculate the area of the second box.
        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
        # Calculate the area of union.
        union_area = box1_area + box2_area - inter_area
        # Calculate the IOU.
        iou = inter_area / union_area if union_area > 0 else 0

        return iou  # Return the IOU.

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Perform non-maximum suppression to filter out overlapping
        bounding boxes.

        Parameters:
        - filtered_boxes (numpy.ndarray):
        Array of shape (?, 4) containing all
          filtered bounding boxes.
        - box_classes (numpy.ndarray):
        Array of shape (?,) containing the class
          number predicted for each box.
        - box_scores (numpy.ndarray):
        Array of shape (?) containing the box scores
          for each box.

        Returns:
        - Tuple of (box_predictions, predicted_box_classes,
        predicted_box_scores):
          - box_predictions (numpy.ndarray):
          Array of shape (?, 4) containing the
            final predicted bounding boxes after non-max suppression.
          - predicted_box_classes (numpy.ndarray):
          Array of shape (?,) containing
            the class numbers corresponding to the final predicted boxes.
          - predicted_box_scores (numpy.ndarray):
          Array of shape (?) containing
            the scores corresponding to the final predicted boxes.
        """
        # Check if there are no filtered boxes
        if len(filtered_boxes) == 0:
            return np.array([]), np.array([]), np.array([])  # empty arrays

        unique_classes = np.unique(box_classes)
        # Get unique classes in box_classes
        box_predictions = []
        # List to store final predicted bounding boxes
        predicted_box_classes = []
        # List to store class numbers for final predicted boxes
        predicted_box_scores = []
        # List to store scores for final predicted boxes

        for cls in unique_classes:  # Iterate over unique classes
            class_indices = np.where(box_classes == cls)[0]
            # Get indices for the current class
            class_boxes = filtered_boxes[class_indices]
            # Get boxes for the current class
            class_scores = box_scores[class_indices]
            # Get scores for the current class
            sorted_indices = np.argsort(class_scores)[::-1]
            # Sort indices in descending order of scores

            while len(sorted_indices) > 0:  # Continue until no more indices
                best_index = sorted_indices[0]
                # Get the index with the highest score
                box_predictions.append(class_boxes[best_index])
                # Append the box to predictions
                predicted_box_classes.append(cls)
                # Append the class to predicted_box_classes
                predicted_box_scores.append(class_scores[best_index])
                # Append the score to predicted_box_scores
                sorted_indices = sorted_indices[1:]  # Remove the best index
                iou_scores = [self.iou(class_boxes[best_index], class_boxes[i])
                              for i in sorted_indices]  # Calculate IOUs
                overlapping_indices = np.where(
                    np.array(iou_scores) > self.nms_t)[0]
                # Get indices of overlapping boxes
                sorted_indices = np.delete(sorted_indices, overlapping_indices)
                # Remove overlapping indices

        box_predictions = np.array(box_predictions)
        # Convert to numpy array
        predicted_box_classes = np.array(predicted_box_classes)
        # Convert to numpy array
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Load images from a folder and return them as numpy arrays.

        Parameters:
        - folder_path (str): Path to the folder containing the images.

        Returns:
        - Tuple of (images, image_paths):
            - images: List of images as numpy.ndarrays.
            - image_paths: List of paths to the individual images in images.
        """

        # Get a list of all image filenames in the folder
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(image) for image in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess a list of images.

        Parameters:
        - images (list of numpy.ndarrays): List of images to preprocess.

        Returns:
        - Tuple of (pimages, image_shapes):
            - pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
                        containing all of the preprocessed images.
            - image_shapes: numpy.ndarray of shape (ni, 2) containing the
                            original height and width of the images.
        """
        # Initialize lists to store preprocessed images and their shapes
        pimages = []
        image_shapes = []

        # Define the input height and width for the Darknet model
        input_h = self.model.input.shape[1].value
        input_w = self.model.input.shape[2].value

        # Preprocess each image
        for image in images:
            # Resize the image with inter-cubic interpolation
            resized_image = cv2.resize(
                image, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            # Rescale the image to have pixel values in the range [0, 1]
            rescaled_image = resized_image / 255.0
            # Append the rescaled image to the list of preprocessed images
            pimages.append(rescaled_image)
            # Append the original shape of the image to the list of image
            # shapes
            image_shapes.append(image.shape[:2])

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Display the image with all boundary boxes, class names, and box scores.

        Parameters:
        - image: a numpy.ndarray containing an unprocessed image
        - boxes: a numpy.ndarray containing the boundary boxes for the image
        - box_classes: a numpy.ndarray containing the class indices foreachbox
        - box_scores: a numpy.ndarray containing the box scores for each box
        - file_name: the file path where the original image is stored
        """
        # Define the color of the bounding boxes
        box_color = (0, 255, 0)  # Blue
        text_color = (0, 0, 255)  # Red
        line_type = cv2.LINE_AA  # Anti-aliased line

        # Iterate over each box
        for box, box_class, box_score in zip(boxes, box_classes, box_scores):
            # Convert the box coordinates to integers
            start_x, start_y, end_x, end_y = map(int, box)

            # Draw the bounding box on the image
            cv2.rectangle(image, (start_x, start_y),
                          (end_x, end_y), box_color, 2)

            # Get the class name and round the box score to 2 decimal places
            class_name = self.class_names[box_class]
            box_score = round(box_score, 2)

            # Define the text to be written on the image
            text = f"{class_name}: {box_score}"

            # Calculate the position of the text
            text_size, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = start_x
            text_y = start_y - text_size[1] - 5

            # Draw the text on the image
            cv2.putText(
                image,
                text,
                (text_x,
                 text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                line_type)

        # Display the image
        cv2.imshow(file_name, image)

        # Wait for a key press and save the image if the 's' key is pressed
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            # Create the directory if it doesn't exist
            if not os.path.exists('detections'):
                os.makedirs('detections')

            # Save the image
            cv2.imwrite(os.path.join('detections', file_name), image)

        # Close the image window
        cv2.destroyAllWindows()

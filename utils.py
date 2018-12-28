import os
import json
import math
import cv2
import numpy as np
from configparser import ConfigParser


def config(section, file='configs.ini'):
    """
        Parse a section of the configs.ini file to obtain settings.

        Args:
            section: A String that defines what section of settings to retrieve
                file: File to search for configs. Default is 'configs.ini'

        Returns:
            dict: A dictionary containing the settings from the specified section
    """
    parser = ConfigParser()
    parser.read(file)

    dict = {}

    # store config settings in a dictionary
    if (parser.has_section(section)):
        params = parser.items(section)

        for param in params:
            dict[param[0]] = param[1]
    else:
        raise Exception('Section {} not found in the {} file'.format(section, file))

    return dict


def get_anchors(anchors_path):
    """
        Retrieve the anchors for a specific dataset.

        Args:
            anchors_path: A string that defines the path to the anchors text file

        Returns:
            anchors: A numpy array containing the anchors for a specific dataset
    """
    anchors_path = os.path.expanduser(anchors_path)

    with open(anchors_path) as file:
        anchors = file.readlines()
        anchors = [[float(x) for x in anchor.split(',')] for anchor in anchors]
        anchors = np.asarray(anchors, dtype=np.int32)

    return anchors


def get_classes(classes_path):
    """
        Retrieve the classes for a specific dataset.

        Args:
            classes_path: A string that defines the path to the classes text file

        Returns:
            classes: A numpy array containing the classes for a specific dataset
    """
    classes_path = os.path.expanduser(classes_path)

    with open(classes_path) as file:
        classes = file.readlines()
        classes = [label.strip() for label in classes]

    return classes


def get_colors(classes):
    """
        Assign a color to each class which will be the color of the bounding box.

        Args:
            classes: A list of the classes used for classifcation

        Returns:
            colors: A list of rgb colors such that each class is assigned a bounding box color
    """
    colors = []
    np.random.seed(0)

    for idx in range(len(classes)):
        r, g, b = tuple(np.random.randint(0, 255, 3))
        colors.append((r, g, b))

    return np.asarray(colors, dtype=np.int16)


def draw_bounding_boxes(image, bboxes, classes, scores, class_names, colors):
    """
        Draw the bounding boxes and their corresponding label onto the image.

        Args:
            image: A numpy array of the image
            bboxes: A list of tuples containing bounding box coordinates (xmin, ymin, xmax, ymax)
            classes: A list of class ids (labels) to decide what color the bounding box is
            scores: A list of confidence scores for the predictions
            class_names: A list of classes, strings, that represent different objects
            colors: A list of rgb colors for bounding boxes

        Returns:
            image: An image containing the bounding boxes and associated labels
    """
    height, width, _ = image.shape
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (0, 0, 0)

    for idx, bbox in enumerate(bboxes):
        label = classes[idx]
        score = scores[idx]
        text = class_names[label] + '  ' + score
        color = tuple(reversed(colors[label]))  # cv2 uses bgr rather than rgb

        # correct predicted bounding box coordinate
        xmin = min(0, int(math.round(bbox[0])))
        ymin = min(0, int(math.round(bbox[1])))
        xmax = max(height, int(math.round(bbox[2])))
        ymax = max(width, int(math.round(bbox[3])))

        # add bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 5)

        # add label and score to top of bounding box
        # Note: will break if ymin is at top of image or xmin is at right of image
        label_size, _ = cv2.getTextSize(text, font_face, font_scale, font_thickness)
        label_xmin = xmin
        label_ymin = ymin - label_size[0] - 6
        label_xmax = xmin + label_size[0] + 6
        label_ymax = ymin

        cv2.rectangle(image, (label_xmin, label_ymin), (label_xmax, label_ymax), color, cv2.CV_FILLED)
        cv2.putText(image, text, (xmin + 3, ymin - 3), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return image


class LoadData:
    """
        Iterator for loading images and annotations into numpy arrays.
    """
    def __init__(self, dataset_info, model_info):
        self.dataset_info = dataset_info
        self.model_info = model_info
        self.annotations_path = os.path.expanduser(self.dataset_info["annotations_path"])
        self.train_path = os.path.expanduser(self.dataset_info["train_path"])
        self.epoch = 1

    def __iterator__(self):
        return self

    def next(self):
        if (self.epoch <= self.model_info["epochs"]):
            pass
        else:
            raise StopIteration

    def load_data(self):
        """
            Load the images and annotations for a specific dataset.

            Args:
                dataset_info: Holds the paths to the images and annotations

            Returns:
                images:
                annotations:
        """
        pass

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
        # Note: may break if ymin is at top of image or xmin is at right of image
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
    def __init__(self, image_path, data_path, num_images, height, width, max_epoch, batch_size):
        self.image_path = image_path
        self.data_path = data_path
        self.num_images = num_images
        self.height = height
        self.width = width
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.max_batch = int(math.ceil(num_images / float(batch_size))
        self.epoch = 0
        self.batch = 0
        self.file = open(data_path, "r")

    def __iterator__(self):
        return self

    def next(self):
        if (self.epoch <= self.max_epoch):
            if(self.batch <= self.max_batch):
                image_names, labels = self.read_file()
                images = self.get_images(image_names)

                return (images, labels)
            else:
                self.epoch += 1
                self.batch = 0
                self.file.seek(0)
        else:
            self.file.close()
            raise StopIteration

    def read_file(self):
        """
            Read the image paths and labels for bounding boxes and categories from a file.

            Returns:
                image_names: A list of paths to image files
                labels: A numpy array containing bounding boxes coordinates and category ids
        """
        image_names = []
        labels = []

        for _ in range(self.batch_size):
            num_bboxes = int(self.file.readline())

            # end of file was reached
            if not num_bboxes:
                break

            image_name = self.file.readline()
            image_names.append(image_name)

            label = []

            # get all the bounding boxes and labels corresponding to the image
            for idx in range(num_bboxes):
                line = self.file.readline()
                data = [float(x) for x in line.split(',')]
                label.append(data)

            labels.append(label)

        return image_names, labels

    def get_images(self, image_names):
        """
            Get a batch of images from the image directory.

            Args:
                image_names: A list of paths to image files

            Returns:
                images: A numpy array containing a batch of images
        """
        images = []

        for image_name in image_names:
            image = cv2.imread(image_name)
            images.append(image)

        images = np.stack(images)

        return images

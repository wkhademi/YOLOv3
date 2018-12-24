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

import os
import numpy as np
import tensorflow as tf
from utils import LoadData

class Train:
    def __init__(self, conf, dataset_info, model_info, anchors, classes, colors):
        self.conf = conf
        self.dataset_info = dataset_info
        self.model_info = model_info
        self.anchors = anchors
        self.classes = classes
        self.colors = colors

        # setup dataset configuration variables
        self.height = int(dataset_info["height"])
        self.width = int(dataset_info["weight"])
        self.channels = int(dataset_info["channels"])
        self.image_path = os.path.expanduser(dataset_info["train_path"])
        self.data_path = os.path.expanduser(dataset_info["data_path"])
        self.num_images = len(os.listdir(self.image_path))

        # setup model configuration variables
        self.num_epochs = int(model_info["epochs"])
        self.batch_size = int(model_info["batch_size"])

    def pretrain(self):
        pass

    def train(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass

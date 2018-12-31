import os
import numpy as np
import tensorflow as tf
from utils import LoadData

class Train:
    def __init__(self, conf, anchors, classes, colors):
        self.conf = conf
        self.anchors = anchors
        self.classes = classes
        self.colors = colors

        # setup dataset configuration variables
        self.height = conf.dataset_info["height"]
        self.width = conf.dataset_info["weight"]
        self.channels = conf.dataset_info["channels"]
        self.image_path = os.path.expanduser(conf.dataset_info["train_path"])
        self.data_path = os.path.expanduser(conf.dataset_info["data_path"])

        # setup model configuration variables
        self.num_epochs = conf.model_info["epochs"]
        self.batch_size = conf.model_info["batch_size"]

    def pretrain(self):
        pass

    def train(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass

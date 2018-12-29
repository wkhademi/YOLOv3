import sys
import argparse
import numpy as np
import tensorflow as tf
from utils import config, get_anchors, get_classes, get_colors, LoadData
from prepare import PrepareData
from display import display_image, display_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='COCO')
    parser.add_argument('--prepdata', type=bool, default=False)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--livefeed', type=bool, default=False)
    conf = parser.parse_args()

    if (conf.pretrain is True and conf.train is True):
        print("Flags [--pretrain] and [--train] cannot be set to true at the same time.")
        sys.exit()

    dataset_info = config(conf.data)
    model_info = config('MODEL')

    # Preprocess data by resizing images and bounding boxes and performing data augmentation
    if (conf.prepdata is True):
        datagen_iterator = PrepareData(dataset_info, model_info)
        datagen = iter(datagen_iterator)
        print("Beginning preprocessing data...")

        while True:
            try:
                next(datagen)
                print("Batch {} of {} ({} of {} images)".format(
                    datagen_iterator.batch, datagen_iterator.max_batch,
                    datagen_iterator.image_num, datagen_iterator.length))
            except StopIteration as e:
                print("Finished preprocessing data.")
                break

    # get info needed for predicting and drawing bounding boxes and object classifcation
    anchors = get_anchors(dataset_info["anchor_path"])
    classes = get_classes(dataset_info["class_path"])
    colors = get_colors(classes)

    if (conf.pretrain is True):
        pass
    elif (conf.train is True):
        datagen_iterator = LoadData(dataset_info, model_info)
    elif (conf.livefeed is True):
        pass
    else:
        pass

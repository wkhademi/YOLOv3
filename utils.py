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


def convert_to_xyxy(xywh):
    """
        Convert a bounding box with center point (x,y), width w, and height h to a
        bounding box with top left coordinate (x1,y1) and bottom right coordinate (x2,y2).

        Args:
            xywh: A list of bounding box info containing x,y,width,height

        Returns:
            xyxy: A list of bounding box info containing x1,y1,x2,y2
    """
    x = xywh[:, 0]
    y = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]

    # convert to (x,y) coordinates
    x1 = x - w
    y1 = y - h
    x2 = x + w
    y2 = y + h

    xyxy = np.asarray([x1, y1, x2, y2])
    xyxy = np.transpose(xyxy, (1, 0))

    return xyxy


def convert_to_xywh(xyxy):
    """
        Convert a bounding box with top left coordinate (x1,y1) and bottom right coordinate
        (x2,y2) to a bounding box with center point (x,y), width w, and height h.

        Args:
            xyxy: A list of bounding box info containing x1,y1,x2,y2

        Returns:
            xywh: A list of bounding box info containing x,y,width,height
    """
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    # get width and height from center of bounding box to edge
    w = (x2 - x1) // 2
    h = (y2 - y1) // 2

    # get center point
    x = x1 + w
    y = y1 + h

    xywh = np.asarray([x, y, w, h])
    xywh = np.transpose(xywh, (1, 0))

    return xywh


def nms(bboxes, scores, max_boxes=25, iou_thresh=0.5):
    """
        Perform non-maximum suppression.

        Arguments:
            bboxes: YOLOv3 models predicted bounding boxes
            scores: Confidence scores for each bounding box
            max_boxes: Max number of bounding box for an image
            iou_threshold: The overlap threshold of bounding boxes

        Returns:
            picked: The indices of the bounding boxes that weren't suppressed
    """
    # get (x, y) coordinates for bounding boxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # calculate the area of each bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort indices of scores in descending order of score
    order = scores.argsort()[::-1]

    # suppress the boxes that have very small IoU with other boxes
    picked = []
    while (order.size > 0 and len(picked) < max_boxes):
        idx = order[0]
        picked.append(idx)

        # get intersection coordinates to calculate IoU
        xx1 = np.maximum(x1[idx], x1[1:])
        yy1 = np.maximum(y1[idx], y1[1:])
        xx2 = np.minimum(x2[idx], x2[1:])
        yy2 = np.mimimum(y2[idx], y2[1:])

        # calculate IoU
        intersection = np.maximum(0., (xx2 - xx1 + 1)) * np.maximum(0., (yy2 - yy1 + 1))
        union = area[idx] + area[1:] - intersection
        iou = intersection / union

        # find next bounding box that wasn't suppressed
        index = np.where(iou <= iou_thresh)[0]
        order = order[index + 1] # add 1 because of 0 index

    return picked


def bounding_box_nms(bboxes, scores, num_classes, max_boxes=25, score_thresh=0.5, iou_thresh=0.5):
    """
        Perform non-maximum suppression on bounding boxes to not have multiple
        bounding boxes for the same object.

        Arguments:
            bboxes: YOLOv3 models predicted bounding boxes
            scores: Confidence scores for each bounding box
            num_classes: Number of classes in dataset
            max_boxes: Max number of bounding box for an image
            score_threshold: The necessary score that a bounding box needs to meet to be kept
            iou_threshold: The overlap threshold of bounding boxes

        Returns:
            bboxes: Predicted bounding boxes that weren't suppressed
            scores: Confidence scores of the bounding boxes that were kept
            labels: The label associated with the bounding box
    """
    kept_bboxes, kept_scores, kept_labels = [], []

    for idx in range(num_classes):
        indices = np.where(scores[:, idx] >= score_thresh)

        # only apply nms to bounding boxes that are above score threshold
        nms_bboxes = bboxes[indices]
        nms_scores = scores[:, idx][indices]

        if not nms_bboxes:
            continue

        # get indices of bounding boxes that aren't suppressed by non-maximum suppression
        indices = nms(nms_bboxes, nms_scores, max_boxes, iou_thresh)

        kept_bboxes.append(nms_bboxes[indices])
        kept_scores.append(nms_scores[indices])
        kept_labels.append(np.ones(len(indices), dtype='int32') * idx)

    if not kept_bboxes:
        return None, None, None

    # put data back into proper format
    bboxes = np.concatenate(kept_bboxes, axis=0)
    scores = np.concatenate(kept_scores, axis=0)
    labels = np.concatenate(kept_labels, axis=0)

    return bboxes, scores, labels


def bounding_box_IoU(pred_bboxes, truth_bboxes):
    """
        Perform Intersection over Union with predicted bounding boxes and ground
        truth bounding boxes to obtain a score of how well model places bounding
        boxes around an object.

        Arguments:
            pred_bboxes: YOLOv3 models predicted bounding boxes
            truth_bboxes: Ground truth bounding boxes

        Returns:
            iou: The Intersection over Union of the predicted bounding boxes and ground truth ones
    """
    # get (x,y) coordinates of predicted and ground truth bounding boxes
    pred_x1, truth_x1 = pred_bboxes[:, 0], truth_bboxes[:, 0]
    pred_y1, truth_y1 = pred_bboxes[:, 1], truth_bboxes[:, 1]
    pred_x2, truth_x2 = pred_bboxes[:, 2], truth_bboxes[:, 2]
    pred_y2, truth_y2 = pred_bboxes[:, 3], truth_bboxes[:, 3]

    # get areas of the predicted and ground truth boxes
    pred_area = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)
    truth_area = (truth_x2 - truth_x1 + 1) * (truth_y2 - truth_y1 + 1)

    # get (x,y) coordinates needed to calculate intersection
    xx1 = np.maximum(pred_x1, truth_x1)
    yy1 = np.maximum(pred_y1, truth_y1)
    xx2 = np.minimum(pred_x2, truth_x2)
    yy2 = np.minimum(pred_y2, truth_y2)

    # calculate IoU
    intersection = np.maximum(0., (xx2 - xx1 + 1)) * np.maximum(0., (yy2 - yy1 + 1))
    union = pred_area + truth_area - intersection
    iou = intersection / union

    return iou


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
        # NOTE: may break if ymin is at top of image or xmin is at right of image
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
        self.max_batch = int(math.ceil(num_images / float(batch_size)))
        self.epoch = 0
        self.batch = 0
        self.file = open(data_path, "r")

    def __iterator__(self):
        return self

    def next(self):
        if (self.epoch <= self.max_epoch):
            if (self.batch <= self.max_batch):
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

            # NOTE: Probably not proper formatting returned. NEED TO FIX...
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

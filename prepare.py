import io
import os
import math
import json
import cv2
import numpy as np

class PrepareData:
    """
        Iterator for preprocessing images and annotations.
    """
    def __init__(self, dataset_info, model_info):
        self.dataset_info = dataset_info
        self.model_info = model_info
        self.annotations_path = os.path.expanduser(self.dataset_info["annotations_path"])
        self.train_path = os.path.expanduser(self.dataset_info["train_path"])
        self.length = len(os.listdir(self.train_path))
        self.image_num = 0
        self.batch = 0
        self.max_batch = int(math.ceil(self.length / float(self.model_info["batch_size"])))

        # parse annotations file for image names and image related data
        with io.open(self.annotations_path, encoding='utf-8') as json_file:
            data = json.loads(json_file.read())
            annotations = data["annotations"]
            self.image_info = {}

            for annotation in annotations:
                image_name = '%012d.jpg' % annotation["image_id"]
                image_path = os.path.join(self.train_path, image_name)
                bbox = annotation["bbox"]
                bbox = {'xmin': bbox[0], 'ymin': bbox[1], 'xmax': bbox[0]+bbox[2], 'ymax': bbox[1]+bbox[3]}
                label = annotation["category_id"]

                # correct for removed classes from category list
                if (label >= 1 and label <= 11):
                    label -= 1
                elif (label >= 13 and label <= 25):
                    label -= 2
                elif (label >= 27 and label <= 28):
                    label -= 3
                elif (label >= 31 and label <= 44):
                    label -= 5
                elif (label >= 46 and label <= 65):
                    label -= 6
                elif (label == 67):
                    label -= 7
                elif (label == 70):
                    label -= 9
                elif (label >= 72 and label <= 82):
                    label -= 10
                elif (label >= 84 and label <= 90):
                    label -= 11

                try:
                    self.image_info[image_path].append([bbox, label])
                except KeyError:
                    self.image_info[image_path] = [[bbox, label]]

    def __iter__(self):
        return self

    def next(self):
        self.batch += 1
        if (self.batch <= self.max_batch):
            if (self.image_num + int(self.model_info["batch_size"]) > self.length):
                increment = self.length - self.image_num
            else:
                increment = int(self.model_info["batch_size"])

            self.load_data(self.image_num, self.image_num + increment)
            self.resize_images(int(self.dataset_info["height"]), int(self.dataset_info["width"]))
            self.resize_bounding_boxes(self.image_num, self.image_num + increment)
            self.save_images(self.image_num, self.image_num + increment)
            self.image_num += increment
        else:
            raise StopIteration

    def load_data(self, start, finish):
        """
            Load the images from the dataset as a numpy array.

            Args:
                start: starting index in set of images
                finish: ending index in set of images
        """
        images = []
        keys = self.image_info.keys()
        for image_path in keys[start:finish]:
            image = cv2.imread(image_path)
            images.append(image)

        self.images = np.asarray(images)

    def resize_images(self, max_height, max_width):
        """
            Downsample all images to a specific max height and width.

            Args:
                max_height: desired output height of image
                max_width: desired output width of image
        """
        self.scales = []
        self.horizontal_shifts = []
        self.vertical_shifts = []

        for idx, image in enumerate(self.images):
            height, width, _ = image.shape

            if (max_height < height or max_width < width):
                scaling_factor = min(max_height / float(height), max_width / float(width))
                self.scales.append(scaling_factor)

                # resize image
                self.images[idx] = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

                # pad images so they are 416x416
                height, width, _ = self.images[idx].shape
                longest_edge = max(height, width)
                top, bottom, left, right = (0, 0, 0, 0)
                BLACK = [0, 0, 0]

                if (height < longest_edge):
                    dh = longest_edge - height
                    top = dh // 2
                    bottom = dh - top
                    self.vertical_shifts.append(top)
                    self.horizontal_shifts.append(left)
                elif (width < longest_edge):
                    dw = longest_edge - width
                    left = dw // 2
                    right = dw - left
                    self.vertical_shifts.append(top)
                    self.horizontal_shifts.append(left)
                else:
                    self.vertical_shifts.append(top)
                    self.horizontal_shifts.append(left)

                self.images[idx] = cv2.copyMakeBorder(self.images[idx], top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
            else:
                top, left = (0, 0)
                scaling_factor = 1
                self.scales.append(scaling_factor)
                self.vertical_shifts.append(top)
                self.horizontal_shifts.append(left)

    def resize_bounding_boxes(self, start, finish):
        """
            Resize the labeled bounding boxes to match the downsampled images.

            Args:
                start: starting index in set of images
                finish: ending index in set of images
        """
        keys = self.image_info.keys()
        for idx, image_path in enumerate(keys[start:finish]):
            height, width, _ = self.images[idx].shape
            image_info = self.image_info[image_path]
            scaling_factor = self.scales[idx]

            # scale and shift all bounding boxes
            for index, data in enumerate(image_info):
                bbox = data[0]
                self.image_info[image_path][index][0]["xmin"] = int(np.round(bbox["xmin"] * scaling_factor)) + self.horizontal_shifts[idx]
                self.image_info[image_path][index][0]["ymin"] = int(np.round(bbox["ymin"] * scaling_factor)) + self.vertical_shifts[idx]
                self.image_info[image_path][index][0]["ymax"] = int(np.round(bbox["ymax"] * scaling_factor)) + self.vertical_shifts[idx]
                self.image_info[image_path][index][0]["xmax"] = int(np.round(bbox["xmax"] * scaling_factor)) + self.horizontal_shifts[idx]

    def save_images(self, start, finish):
        """
            Save resized images back to dataset directory and save all the images info
            to a separate file.

            Args:
                start: starting index in set of images
                finish: ending index in set of images
        """
        with open(self.dataset_info["data_path"], 'a') as bbox_id_file:
            keys = self.image_info.keys()
            for idx, image_path in enumerate(keys[start:finish]):
                image = self.images[idx]
                image_info = self.image_info[image_path]

                # save resized image
                cv2.imwrite(image_path, image)

                # write the number of bounding boxes for the image to file
                bbox_id_file.write(str(len(image_info)) + '\n')

                # write image path to file
                bbox_id_file.write(image_path + '\n')

                # write bounding boxes and categories to a file
                for index, bbox in enumerate(image_info):
                    xmin = str(bbox[0]["xmin"])
                    ymin = str(bbox[0]["ymin"])
                    xmax = str(bbox[0]["xmax"])
                    ymax = str(bbox[0]["ymax"])
                    category_id = str(bbox[1])
                    bbox_id_file.write(xmin + ',' + ymin +',' + xmax + ',' + ymax + ',' + category_id + '\n')

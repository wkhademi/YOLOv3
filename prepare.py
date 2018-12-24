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
            if (self.image_num + int(self.model_info["batch_size"]) > self.length): # partial batch
                self.load_data(self.image_num, self.length)
                self.image_num += (self.length - self.image_num)
            else: # full batch
                self.load_data(self.image_num, self.image_num + int(self.model_info["batch_size"]))
                self.image_num += int(self.model_info["batch_size"])

            self.resize_images(int(self.dataset_info["height"]), int(self.dataset_info["width"]))
        else:
            raise StopIteration


    def load_data(self, start, finish):
        """
            Load the images from the dataset as a numpy array.
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
        """
        for idx, image in enumerate(self.images):
            height, width, _ = image.shape

            if max_height < height or max_width < width:
                scaling_factor = max_height / float(height)

                if (max_width / float(width) < scaling_factor):
                    scaling_factor = max_width / float(width)

                self.images[idx] = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)


    def preprocess_images(self):
        """
            Perform random data augmentation on image set.
        """
        pass


    def resize_bounding_boxes(self):
        """
            Resize the labeled bounding boxes to match the downsampled images.
        """
        pass

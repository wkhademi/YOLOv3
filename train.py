import os
import numpy as np
import tensorflow as tf
from utils import LoadData
from yolov3 import YOLOv3

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
        """
            trains YOLOv3 model
        """
        x = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels), name="inputs")
        y = tf.placeholder(tf.float32, shape=(None, self.height, self.width, 3, 5+len(self.classes)), name="outputs")

        datagen_iterator = LoadData(self.image_path, self.data_path, self.num_images, self.height, self.width, self.num_epochs, self.batch_size)
        datagen = iter(datagen_iterator)

        yolov3 = YOLOv3()
        yolov3.build_model()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_list = []

            while True:
                try:
                    images, labels, finished_epoch = next(datagen)
                    loss, _ = sess.run([yolov3.loss, yolov3.optimizer], feed_dict={x: images, y: labels, is_training: True})
                    loss_list.append(loss)

                    # calculate average loss over single epoch
                    if finished_epoch:
                        average_loss = np.mean(loss_list)
                        print("Average loss: " + str(average_loss))
                        loss_list = []

                except StopIteration:
                    print("Finished training")
                    break

    def load_model(self):
        pass

    def save_model(self):
        pass

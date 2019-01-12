# YOLOv3
YOLOv3 object detection model for Cal Poly Robotics Club autonomous golf cart.

## Packages Needed
The packages needed to run the YOLOv3 object detection model include:
- configparser
- matplotlib
- numpy
- opencv-python
- tensorflow

## Dataset
The current dataset being used to train our YOLOv3 objection detection model is the COCO2017 dataset.  
The dataset can be downloaded from the [COCO Website](http://cocodataset.org/#home).

## Preprocessing Data
To preprocess the COCO dataset, and other datasets matching the COCO dataset format, run the command:  
`python main.py --prepdata=True`.  

By running this command the original images and annotations of the COCO dataset will be parsed and processed so that the images are resized to 416x416 and a file is created containing the image names, bounding box coordinates, and categories of the bounding boxes.

*Note:* Data only needs to be preprocessed one time. It is advised to run this command as soon as the images and annotations of the COCO data have first been downloaded.

## Pre-Trained Weights
Pre-trained weights for the YOLOv3 model can be downloaded from the [YOLO Website](https://pjreddie.com/darknet/yolo/).  
To use the pre-trained weights with the YOLOv3 model, run the command: `python main.py --pretrain=True`

## Training the Model
To be added...

## Testing the Model
To be added...

## Testing the Model on Live Camera Data
To be added...

## Original Paper
The original YOLOv3 paper was written by Joseph Redmon and Ali Farhadi.  
The paper can be found here: [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf).

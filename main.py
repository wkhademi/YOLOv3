import argparse
import numpy as np
import tensorflow as tf
from utils import config, get_anchors, get_classes

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, default='COCO')
	parser.add_argument('--pre_train', type=bool, default=False)
	parser.add_argument('--train', type=bool, default=True)
	conf = parser.parse_args()

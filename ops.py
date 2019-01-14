import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def convolution_2d(inputs,
                   weight_shape,
                   bias_shape,
                   stride=3,
                   padding='SAME',
                   weight_init=tf.glorot_uniform_initializer(),
                   bias_init=tf.constant_initializer(0.0),
                   batch_norm=True,
                   activation=tf.nn.relu,
                   is_training=True,
                   scope=None):
    pass


def residual_block(inputs,
                   num_layers,
                   weight_shapes,
                   bias_shapes,
                   strides,
                   weight_inits,
                   bias_inits,
                   batch_norms,
                   activations,
                   is_training=True,
                   scope=None):
    pass


def activation_fn(inputs,
                  fn=tf.nn.relu,
                  scope=None):
    pass


def pooling(inputs,
            ksize=2,
            stride=2,
            padding='SAME',
            scope=None):
    pass


def upsample(inputs,
             ksize,
             factor,
             scope=None):
    # Needs more arguments. Too lazy to finish right now...
    pass


def bilinear_upsample_filter(ksize,
                             factor,
                             in_channels,
                             out_channels):
    """
        Create a 2D bilinear kernel used to upsample.

        Arguments:
            ksize:
            factor:
            in_channels:
            out_channels:

        Returns:
            weights:
    """
    if (ksize % 2 == 1):
        center = factor - 1
    else:
        center = factor - 0.5

    # bilinear filter
    og = np.ogrid[:ksize, :ksize]
    kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    # apply bilinear filter to proper kernel size
    weights = np.zeros((in_channels, out_channels, ksize, ksize), dtype=np.float32)
    weights[range(in_channels), range(out_channels), : , :] = kernel
    weights = np.transpose(weights, (2, 3, 0, 1))

    return weights

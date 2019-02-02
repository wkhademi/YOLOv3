import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def convolution_2d(inputs,
                   input_channels,
                   output_channels,
                   weight_shape,
                   bias_shape,
                   stride=3,
                   padding='SAME',
                   weight_init=tf.glorot_uniform_initializer(),
                   bias_init=tf.constant_initializer(0.0),
                   batch_norm=True,
                   is_training=True,
                   activation=tf.nn.relu,
                   scope=None):
    """
        Convolution layer.

        Args:
            inputs: A Numpy array containing the inputs to be convolved
            in_channels: The number of channels in inputs
            out_channels: The desired number of channels for the outputs
            weight_shape: The desired size of the filters
            bias_shape: The desired size of the bias vector
            stride: Amount to slide filter over by when convolving with inputs
            padding: "SAME" or "VALID"
            weight_init: The initialized values for the filters
            bias_init: The initialized values for the bias
            batch_norm: A bool denoting whether or not to perfom batch normalization
            is_training: A bool denoting whether is training or test time
            activation: A TensorFlow function that applies a non-linear activation
            scope: A str denoting the scope for the convolution layer

        Returns:
            conv: A Numpy array that has been convolved with a set of filters and
                  possibly had batch normalization and a non-linear activation applied to it
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weights", shape=[weight_shape, weight_shape, in_channels, out_channels],
                                  dtype=tf.float32, initializer=weight_init)

        conv = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1], padding, 'conv2d')

        biases = tf.get_variable("biases", shape=[bias_shape], dtype=tf.float32,
                                 initializer=bias_init)

        conv = tf.nn.bias_add(conv, biases, 'conv2d_preact')

        if (batch_norm):
            conv = tf.layers.batch_normalization(conv, training=is_training)

        if (activation):
            conv = activation(conv, 'conv2d_act')

    return conv


def upsample(inputs,
             in_channels,
             out_shape,
             out_channels,
             ksize,
             factor,
             padding="SAME"
             scope=None):
    """
        Upsample inputs by some factor.

        Args:
            inputs: A Numpy array containing the inputs to be upsampled
            in_channels: The number of channels in inputs
            out_shape: The desired height and width for the outputs
            out_channels: The desired number of channels for the outputs
            ksize: The desired size of the filters to be used during upsampling
            factor: The amount to upsample the inputs by
            padding: "SAME" or "VALID"
            scope: A str denoting the scope for the upsampling layer

        Returns:
            upsample: A Numpy array containing the upsampled inputs
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        weights = bilinear_upsample_filter(ksize, factor, in_channels, out_channels)

        upsample = tf.nn.conv2d_transpose(inputs, weights, [-1, out_shape, out_shape, out_channels],
                                          [1, factor, factor, 1], padding=padding, name="upsample")

    return upsample


def bilinear_upsample_filter(ksize,
                             factor,
                             in_channels,
                             out_channels):
    """
        Create a 2D bilinear kernel used to upsample.

        Arguments:
            ksize: The desired size of bilinear filter
            factor: The amount you want to scale the input up by
            in_channels: The number of input channels
            out_channels: The desired number of output channels

        Returns:
            weights: A Numpy array containing the set of filters used during upsampling
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
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        pass

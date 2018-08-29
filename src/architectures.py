import numpy as np
import tensorflow as tf

from functools import partial
from tensorflow import (
    orthogonal_initializer as ort_init, zeros_initializer as zeros_init)
from tensorflow.contrib.layers import xavier_initializer as xavier_init
from tensorflow.python.layers.layers import (
    conv2d, conv2d_transpose as deconv2d, dense)
from tensorflow.python.ops.nn import relu
from utils_tf import generator_summaries, leaky_relu, stats_summaries

# some architectures (and hyper-parameters) are drawn from the original work on
# "Boundary-Seeking Generative Adversarial Networks"
# ref: https://github.com/rdevon/BGAN
# The original code was written in Theano/Lasagne and has been ported to TF

# NOTE: conv2d/deconv2d paddings may be slightly different from the ones in
# Lasagne. To make them equivalent, a manual pad2 is necessary. Sketch:
# pad2 = lambda x: tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

# NOTE: this batch normalization is slightly different from the one in Lasagne.
# See the LS-TF porting notes for further info.
batch_norm = partial(tf.layers.batch_normalization, training=True)


# 28x28 GAN architectures (binary)
def _gan28_discr(experiment=None, X=None, reuse=False, **kwargs):
    lrelu = partial(leaky_relu, leakiness=experiment.leak)
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "D_SHAPE {} {} [reuse={}]"
    logger.debug(msg.format("in", X.shape, reuse))
    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("hidden1"):
            d_hidden1 = conv2d(
                X, filters=h_dim, kernel_size=5, strides=2, padding="same",
                activation=lrelu, kernel_initializer=xavier_init())
            logger.debug(msg.format("dh1", d_hidden1.shape, reuse))
        with tf.variable_scope("hidden2"):
            d_hidden2 = conv2d(
                d_hidden1, filters=h_dim * 2, kernel_size=5, strides=2,
                padding="same", activation=lrelu,
                kernel_initializer=xavier_init())
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
        with tf.variable_scope("hidden3"):
            d_hidden2 = tf.reshape(
                d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
            d_hidden3 = dense(d_hidden2, 1024, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh3", d_hidden3.shape, reuse))
        with tf.variable_scope("output"):
            d_out = dense(d_hidden3, 1, kernel_initializer=xavier_init())
            logger.debug(msg.format("out", d_out.shape, reuse))

            # define summaries on the last layer
            d_summaries = stats_summaries(d_out)

    return d_out, d_summaries


def _gan28_gen(experiment=None, z=None, **kwargs):
    # Batch normalization is applied after the layer's activation function and
    # without removing its bias.
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "G_SHAPE {}: {}"
    logger.debug(msg.format("in", z.shape))
    with tf.variable_scope("generator"):
        with tf.variable_scope("hidden1"):
            g_hidden1 = batch_norm(dense(z, 1024, activation=relu))
            logger.debug(msg.format("gh1", g_hidden1.shape))
        with tf.variable_scope("hidden2"):
            g_hidden2 = batch_norm(dense(
                g_hidden1, h_dim * 2 * 7 * 7, activation=relu))
            logger.debug(msg.format("gh2", g_hidden2.shape))
            g_hidden2 = tf.reshape(g_hidden2, [-1, h_dim * 2, 7, 7])
            logger.debug(msg.format("gh2", g_hidden2.shape))
        with tf.variable_scope("hidden3"):
            # deconv2d only supports nhwc channels. Transposing nchw to nhwc
            g_hidden2 = tf.transpose(g_hidden2, [0, 3, 2, 1])
            logger.debug(msg.format("gh3", g_hidden2.shape))
            g_hidden3 = batch_norm(deconv2d(
                g_hidden2, filters=h_dim, kernel_size=5, strides=2,
                padding="same", activation=relu, kernel_initializer=ort_init(),
                bias_initializer=zeros_init()))
            logger.debug(msg.format("gh3", g_hidden3.shape))
        with tf.variable_scope("output"):
            g_out = deconv2d(
                g_hidden3, filters=1, kernel_size=5, strides=2, padding="same",
                kernel_initializer=ort_init(), bias_initializer=zeros_init())
            logger.debug(msg.format("out", g_out.shape))

            # define summaries on the last layer
            g_summaries, images_summaries = generator_summaries(
                g_out, experiment)

    return g_out, g_summaries, images_summaries


def _gan28_gen_no_act_no_bias(experiment=None, z=None, **kwargs):
    # Batch normalization is applied before the layer's activation function and
    # removing its bias.
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "G_SHAPE {}: {}"
    logger.debug(msg.format("in", z.shape))
    with tf.variable_scope("generator"):
        with tf.variable_scope("hidden1"):
            g_hidden1 = relu(batch_norm(dense(
                z, 1024, activation=None, use_bias=False)))
            logger.debug(msg.format("gh1", g_hidden1.shape))
        with tf.variable_scope("hidden2"):
            g_hidden2 = relu(batch_norm(dense(
                g_hidden1, h_dim * 2 * 7 * 7, activation=None,
                use_bias=False)))
            logger.debug(msg.format("gh2", g_hidden2.shape))
            g_hidden2 = tf.reshape(g_hidden2, [-1, h_dim * 2, 7, 7])
            logger.debug(msg.format("gh2", g_hidden2.shape))
        with tf.variable_scope("hidden3"):
            # deconv2d only supports nhwc channels. Transposing nchw to nhwc
            g_hidden2 = tf.transpose(g_hidden2, [0, 3, 2, 1])
            logger.debug(msg.format("gh3", g_hidden2.shape))
            g_hidden3 = relu(batch_norm(deconv2d(
                g_hidden2, filters=h_dim, kernel_size=5, strides=2,
                padding="same", kernel_initializer=ort_init(),
                bias_initializer=zeros_init(), use_bias=False,
                activation=None)))
            logger.debug(msg.format("gh3", g_hidden3.shape))
        with tf.variable_scope("output"):
            g_out = deconv2d(
                g_hidden3, filters=1, kernel_size=5, strides=2, padding="same",
                kernel_initializer=ort_init(), bias_initializer=zeros_init())
            logger.debug(msg.format("out", g_out.shape))

            # define summaries on the last layer
            g_summaries, images_summaries = generator_summaries(
                g_out, experiment)

    return g_out, g_summaries, images_summaries


# 20x20 GAN architectures (binary)
def _gan20_discr(experiment=None, X=None, reuse=False, **kwargs):
    lrelu = partial(leaky_relu, leakiness=experiment.leak)
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "D_SHAPE {} {} [reuse={}]"
    logger.debug(msg.format("in", X.shape, reuse))
    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("hidden1"):
            d_hidden1 = conv2d(
                X, filters=h_dim, kernel_size=5, strides=2, padding="same",
                activation=lrelu, kernel_initializer=xavier_init())
            logger.debug(msg.format("dh1", d_hidden1.shape, reuse))
        with tf.variable_scope("hidden2"):
            d_hidden2 = conv2d(
                d_hidden1, filters=h_dim * 2, kernel_size=5, strides=2,
                padding="same", activation=lrelu,
                kernel_initializer=xavier_init())
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
            d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(
                d_hidden2.shape[1:], dtype=int)])
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
        with tf.variable_scope("hidden3"):
            d_hidden3 = dense(d_hidden2, 1024, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh3", d_hidden3.shape, reuse))
        with tf.variable_scope("output"):
            d_out = dense(d_hidden3, 1, kernel_initializer=xavier_init())
            logger.debug(msg.format("out", d_out.shape, reuse))

            # define summaries on the last layer
            d_summaries = stats_summaries(d_out)

    return d_out, d_summaries


def _gan20_discr_32layer(experiment=None, X=None, reuse=False, **kwargs):
    lrelu = partial(leaky_relu, leakiness=experiment.leak)
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "D_SHAPE {} {} [reuse={}]"
    logger.debug(msg.format("in", X.shape, reuse))
    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("hidden1"):
            d_hidden1 = conv2d(
                X, filters=h_dim, kernel_size=5, strides=2, padding="same",
                activation=lrelu, kernel_initializer=xavier_init())
            logger.debug(msg.format("dh1", d_hidden1.shape, reuse))
        with tf.variable_scope("hidden2"):
            d_hidden2 = conv2d(
                d_hidden1, filters=h_dim * 2, kernel_size=5, strides=2,
                padding="same", activation=lrelu,
                kernel_initializer=xavier_init())
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
            d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(
                d_hidden2.shape[1:], dtype=int)])
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
        with tf.variable_scope("hidden3"):
            d_hidden3 = dense(d_hidden2, 1024, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh3", d_hidden3.shape, reuse))
        with tf.variable_scope("hidden4"):
            d_hidden4 = dense(d_hidden3, 32, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh4", d_hidden4.shape, reuse))
        with tf.variable_scope("output"):
            d_out = dense(d_hidden4, 1, kernel_initializer=xavier_init())
            logger.debug(msg.format("out", d_out.shape, reuse))

            # define summaries on the last layer
            d_summaries = stats_summaries(d_out)

    return d_out, d_summaries


def _gan20_gen(experiment=None, z=None, **kwargs):
    # Batch normalization is applied after the layer's activation function and
    # without removing its bias.
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "G_SHAPE {}: {}"
    logger.debug(msg.format("in", z.shape))
    with tf.variable_scope("generator"):
        with tf.variable_scope("hidden1"):
            g_hidden1 = batch_norm(dense(z, 1024, activation=relu))
            logger.debug(msg.format("gh1", g_hidden1.shape))
        with tf.variable_scope("hidden2"):
            g_hidden2 = batch_norm(dense(
                g_hidden1, h_dim * 2 * 5 * 5, activation=relu))
            logger.debug(msg.format("gh2", g_hidden2.shape))
            g_hidden2 = tf.reshape(g_hidden2, [-1, h_dim * 2, 5, 5])
            logger.debug(msg.format("gh2", g_hidden2.shape))
        with tf.variable_scope("hidden3"):
            # deconv2d only supports nhwc channels. Transposing nchw to nhwc
            g_hidden2 = tf.transpose(g_hidden2, [0, 3, 2, 1])
            logger.debug(msg.format("gh3", g_hidden2.shape))
            g_hidden3 = batch_norm(deconv2d(
                g_hidden2, filters=h_dim, kernel_size=5, strides=2,
                padding="same", activation=relu, kernel_initializer=ort_init(),
                bias_initializer=zeros_init()))
            logger.debug(msg.format("gh3", g_hidden3.shape))
        with tf.variable_scope("output"):
            g_out = deconv2d(
                g_hidden3, filters=1, kernel_size=5,
                strides=2, padding="same", kernel_initializer=ort_init(),
                bias_initializer=zeros_init())
            logger.debug(msg.format("out", g_out.shape))

            # define summaries on the last layer
            g_summaries, images_summaries = generator_summaries(
                g_out, experiment)

    return g_out, g_summaries, images_summaries


# 60x60 GAN architectures (binary)
def _gan60_gen(experiment=None, z=None, **kwargs):
    # Batch normalization is applied after the layer's activation function and
    # without removing its bias.
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "G_SHAPE {}: {}"
    logger.debug(msg.format("in", z.shape))
    with tf.variable_scope("generator"):
        with tf.variable_scope("hidden1"):
            g_hidden1 = batch_norm(dense(z, 1024, activation=relu))
            logger.debug(msg.format("gh1", g_hidden1.shape))
        with tf.variable_scope("hidden2"):
            g_hidden2 = batch_norm(dense(
                g_hidden1, h_dim * 2 * 5 * 5, activation=relu))
            logger.debug(msg.format("gh2", g_hidden2.shape))
            g_hidden2 = tf.reshape(g_hidden2, [-1, h_dim * 2, 5, 5])
            logger.debug(msg.format("gh2", g_hidden2.shape))
        with tf.variable_scope("hidden3"):
            # deconv2d only supports nhwc channels. Transposing nchw to nhwc
            g_hidden2 = tf.transpose(g_hidden2, [0, 3, 2, 1])
            logger.debug(msg.format("gh3", g_hidden2.shape))
            g_hidden3 = batch_norm(deconv2d(
                g_hidden2, filters=h_dim, kernel_size=5, strides=2,
                padding="same", activation=relu, kernel_initializer=ort_init(),
                bias_initializer=zeros_init()))
            logger.debug(msg.format("gh3", g_hidden3.shape))
        with tf.variable_scope("hidden4"):
            g_hidden4 = batch_norm(deconv2d(
                g_hidden3, filters=h_dim, kernel_size=5, strides=2,
                padding="same", activation=relu, kernel_initializer=ort_init(),
                bias_initializer=zeros_init()))
            logger.debug(msg.format("gh4", g_hidden4.shape))
        with tf.variable_scope("output"):
            g_out = deconv2d(
                g_hidden4, filters=1, kernel_size=5,
                strides=3, padding="same", kernel_initializer=ort_init(),
                bias_initializer=zeros_init())
            logger.debug(msg.format("out", g_out.shape))

            # define summaries on the last layer
            g_summaries, images_summaries = generator_summaries(
                g_out, experiment)

    return g_out, g_summaries, images_summaries


def _gan60_discr_32layer(experiment=None, X=None, reuse=False, **kwargs):
    lrelu = partial(leaky_relu, leakiness=experiment.leak)
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "D_SHAPE {} {} [reuse={}]"
    logger.debug(msg.format("in", X.shape, reuse))
    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("hidden1"):
            d_hidden1 = conv2d(
                X, filters=h_dim, kernel_size=5, strides=2, padding="same",
                activation=lrelu, kernel_initializer=xavier_init())
            logger.debug(msg.format("dh1", d_hidden1.shape, reuse))
        with tf.variable_scope("hidden15"):
            d_hidden15 = conv2d(
                d_hidden1, filters=h_dim, kernel_size=5, strides=2,
                padding="same", activation=lrelu,
                kernel_initializer=xavier_init())
            logger.debug(msg.format("dh15", d_hidden15.shape, reuse))
        with tf.variable_scope("hidden2"):
            d_hidden2 = conv2d(
                d_hidden15, filters=h_dim * 2, kernel_size=5, strides=2,
                padding="same", activation=lrelu,
                kernel_initializer=xavier_init())
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
            d_hidden2 = tf.reshape(d_hidden2, [-1, np.prod(
                d_hidden2.shape[1:], dtype=int)])
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
        with tf.variable_scope("hidden3"):
            d_hidden3 = dense(d_hidden2, 1024, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh3", d_hidden3.shape, reuse))
        with tf.variable_scope("hidden4"):
            d_hidden4 = dense(d_hidden3, 32, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh4", d_hidden4.shape, reuse))
        with tf.variable_scope("output"):
            d_out = dense(d_hidden4, 1, kernel_initializer=xavier_init())
            logger.debug(msg.format("out", d_out.shape, reuse))

            # define summaries on the last layer
            d_summaries = stats_summaries(d_out)

    return d_out, d_summaries


# 20x20 CAN architectures (binary)
def _can20_discr_32layer_auto(
        experiment=None, X=None, reuse=False, use_constraints=None,
        constraints_features=None, **kwargs):
    lrelu = partial(leaky_relu, leakiness=experiment.leak)
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "D_SHAPE {} {} [reuse={}]"
    logger.debug(msg.format("in", X.shape, reuse))
    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("hidden1"):
            d_hidden1 = conv2d(
                X, filters=h_dim, kernel_size=5, strides=2, padding="same",
                activation=lrelu, kernel_initializer=xavier_init())
            logger.debug(msg.format("dh1", d_hidden1.shape, reuse))
        with tf.variable_scope("hidden2"):
            d_hidden2 = conv2d(
                d_hidden1, filters=h_dim * 2, kernel_size=5, strides=2,
                padding="same", activation=lrelu,
                kernel_initializer=xavier_init())
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
            d_hidden2 = tf.reshape(
                d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
        with tf.variable_scope("hidden3"):
            d_hidden3 = dense(d_hidden2, 1024, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh3", d_hidden3.shape, reuse))
        with tf.variable_scope("hidden4"):
            d_hidden4 = dense(d_hidden3, 32, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh4", d_hidden4.shape, reuse))
            d_summaries_dh4 = stats_summaries(d_hidden4, "dh4_pre_cond")
        with tf.variable_scope("shared_weights"):
            d_out_kernel = tf.get_variable(
                "d_out_kernel", shape=[32, 1], initializer=xavier_init())
            logger.debug(msg.format("d_out_kernel", d_out_kernel.shape, reuse))
            d_out_bias = tf.get_variable(
                "d_out_bias", shape=[1, 1], initializer=xavier_init())
            logger.debug(msg.format("d_out_bias", d_out_bias.shape, reuse))

        def skip_constraints():
            with tf.variable_scope("output"):
                d_out = tf.add(tf.matmul(d_hidden4, d_out_kernel),
                               d_out_bias, name="d_out_{}".format(reuse))
                logger.debug(msg.format("out", d_out.shape, reuse))

                # define summaries on the last layer
                d_summaries = tf.summary.merge(
                    [stats_summaries(d_out), d_summaries_dh4])
            return d_out, d_summaries

        def apply_constraints():
            logger.debug("Using constraints: {}".format(
                str(experiment.constraints)))
            with tf.variable_scope("constrained_out"):
                d_constraints_kernel = tf.get_variable(
                    "d_constraints_kernel",
                    shape=[constraints_features.shape[1], 1],
                    initializer=xavier_init())
                logger.debug(msg.format(
                    "d_constraints_kernel", d_constraints_kernel.shape, reuse))
                input_concat = tf.concat(
                    [d_hidden4, constraints_features],
                    axis=1, name="input_concat_{}".format(reuse))
                logger.debug(msg.format(
                    "input_concat", input_concat.shape, reuse))
                weight_concat = tf.concat(
                    [d_out_kernel, d_constraints_kernel],
                    axis=0, name="weight_concat_{}".format(reuse))
                logger.debug(msg.format(
                    "weight_concat", weight_concat.shape, reuse))
                d_constrained_out = tf.add(
                    tf.matmul(input_concat, weight_concat), d_out_bias,
                    name="d_constrained_out_{}".format(reuse))
                logger.debug(msg.format(
                    "constrained_out", d_constrained_out.shape, reuse))

                # define summaries on the last layer
                d_summaries = tf.summary.merge(
                    [stats_summaries(d_constrained_out),
                     d_summaries_dh4])
            return d_constrained_out, d_summaries

        return tf.cond(tf.cast(use_constraints, tf.bool),
                       skip_constraints, apply_constraints)


def _can20_discr_last_layer_auto(
        experiment=None, X=None, reuse=False, use_constraints=None,
        constraints_features=None, **kwargs):
    lrelu = partial(leaky_relu, leakiness=experiment.leak)
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "D_SHAPE {} {} [reuse={}]"
    logger.debug(msg.format("in", X.shape, reuse))
    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("hidden1"):
            d_hidden1 = conv2d(
                X, filters=h_dim, kernel_size=5, strides=2, padding="same",
                activation=lrelu, kernel_initializer=xavier_init())
            logger.debug(msg.format("dh1", d_hidden1.shape, reuse))
        with tf.variable_scope("hidden2"):
            d_hidden2 = conv2d(
                d_hidden1, filters=h_dim * 2, kernel_size=5, strides=2,
                padding="same", activation=lrelu,
                kernel_initializer=xavier_init())
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
            d_hidden2 = tf.reshape(
                d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
        with tf.variable_scope("hidden3"):
            d_hidden3 = dense(d_hidden2, 1024, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh3", d_hidden3.shape, reuse))
        with tf.variable_scope("output"):
            d_out = dense(d_hidden3, 1, activation=lrelu,
                          kernel_initializer=xavier_init())
            logger.debug(msg.format("out", d_out.shape, reuse))
            d_out_summaries = stats_summaries(d_out)

        def skip_constraints():
            return d_out, d_out_summaries

        def apply_constraints():
            logger.debug("Using constraints: {}".format(
                str(experiment.constraints)))
            with tf.variable_scope("constrained_ll_out"):
                constraints_out = dense(
                    constraints_features, 1, activation=lrelu,
                    kernel_initializer=xavier_init())

                d_out_kernel = tf.get_variable(
                    "d_out_kernel", shape=[2, 1], initializer=xavier_init())
                logger.debug(
                    msg.format("d_out_kernel", d_out_kernel.shape, reuse))
                d_out_bias = tf.get_variable(
                    "d_out_bias", shape=[1, 1], initializer=xavier_init())
                logger.debug(msg.format("d_out_bias", d_out_bias.shape, reuse))

                input_concat = tf.concat(
                    [d_out, constraints_out],
                    axis=1, name="input_concat_{}".format(reuse))
                logger.debug(msg.format(
                    "input_concat", input_concat.shape, reuse))
                d_constrained_out = tf.add(
                    tf.matmul(input_concat, d_out_kernel), d_out_bias,
                    name="d_constrained_out_{}".format(reuse))
                logger.debug(msg.format(
                    "constrained_out", d_constrained_out.shape, reuse))

                # define summaries on the last layer
                d_summaries = tf.summary.merge(
                    [stats_summaries(d_constrained_out), d_out_summaries])
            return d_constrained_out, d_summaries

        return tf.cond(tf.cast(use_constraints, tf.bool),
                       skip_constraints, apply_constraints)


# 60x60 CAN architectures (binary)
def _can60_discr_32layer_auto(
        experiment=None, X=None, reuse=False, use_constraints=None,
        constraints_features=None, **kwargs):
    lrelu = partial(leaky_relu, leakiness=experiment.leak)
    h_dim = experiment.h_dim
    logger = experiment.logger

    msg = "D_SHAPE {} {} [reuse={}]"
    logger.debug(msg.format("in", X.shape, reuse))
    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("hidden1"):
            d_hidden1 = conv2d(
                X, filters=h_dim, kernel_size=5, strides=2, padding="same",
                activation=lrelu, kernel_initializer=xavier_init())
            logger.debug(msg.format("dh1", d_hidden1.shape, reuse))
        with tf.variable_scope("hidden15"):
            d_hidden15 = conv2d(
                d_hidden1, filters=h_dim, kernel_size=5, strides=2,
                padding="same", activation=lrelu,
                kernel_initializer=xavier_init())
            logger.debug(msg.format("dh15", d_hidden15.shape, reuse))
        with tf.variable_scope("hidden2"):
            d_hidden2 = conv2d(
                d_hidden15, filters=h_dim * 2, kernel_size=5, strides=2,
                padding="same", activation=lrelu,
                kernel_initializer=xavier_init())
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
            d_hidden2 = tf.reshape(
                d_hidden2, [-1, np.prod(d_hidden2.shape[1:], dtype=int)])
            logger.debug(msg.format("dh2", d_hidden2.shape, reuse))
        with tf.variable_scope("hidden3"):
            d_hidden3 = dense(d_hidden2, 1024, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh3", d_hidden3.shape, reuse))
        with tf.variable_scope("hidden4"):
            d_hidden4 = dense(d_hidden3, 32, activation=lrelu,
                              kernel_initializer=xavier_init())
            logger.debug(msg.format("dh4", d_hidden4.shape, reuse))
            d_summaries_dh4 = stats_summaries(d_hidden4, "dh4_pre_cond")
        with tf.variable_scope("shared_weights"):
            d_out_kernel = tf.get_variable(
                "d_out_kernel", shape=[32, 1], initializer=xavier_init())
            logger.debug(msg.format("d_out_kernel", d_out_kernel.shape, reuse))
            d_out_bias = tf.get_variable(
                "d_out_bias", shape=[1, 1], initializer=xavier_init())
            logger.debug(msg.format("d_out_bias", d_out_bias.shape, reuse))

        def skip_constraints():
            with tf.variable_scope("output"):
                d_out = tf.add(tf.matmul(d_hidden4, d_out_kernel),
                               d_out_bias, name="d_out_{}".format(reuse))
                logger.debug(msg.format("out", d_out.shape, reuse))

                # define summaries on the last layer
                d_summaries = tf.summary.merge(
                    [stats_summaries(d_out), d_summaries_dh4])
            return d_out, d_summaries

        def apply_constraints():
            logger.debug("Using constraints: {}".format(
                str(experiment.constraints)))
            with tf.variable_scope("constrained_out"):
                d_constraints_kernel = tf.get_variable(
                    "d_constraints_kernel",
                    shape=[constraints_features.shape[1], 1],
                    initializer=xavier_init())
                logger.debug(msg.format(
                    "d_constraints_kernel", d_constraints_kernel.shape, reuse))
                input_concat = tf.concat(
                    [d_hidden4, constraints_features],
                    axis=1, name="input_concat_{}".format(reuse))
                logger.debug(msg.format(
                    "input_concat", input_concat.shape, reuse))
                weight_concat = tf.concat(
                    [d_out_kernel, d_constraints_kernel],
                    axis=0, name="weight_concat_{}".format(reuse))
                logger.debug(msg.format(
                    "weight_concat", weight_concat.shape, reuse))
                d_constrained_out = tf.add(
                    tf.matmul(input_concat, weight_concat), d_out_bias,
                    name="d_constrained_out_{}".format(reuse))
                logger.debug(msg.format(
                    "constrained_out", d_constrained_out.shape, reuse))

                # define summaries on the last layer
                d_summaries = tf.summary.merge(
                    [stats_summaries(d_constrained_out),
                     d_summaries_dh4])
            return d_constrained_out, d_summaries

        return tf.cond(tf.cast(use_constraints, tf.bool),
                       skip_constraints, apply_constraints)

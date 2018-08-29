import os
import tensorflow as tf

from inspect import getmembers, isbuiltin, ismethod
from tensorflow.python.client import device_lib


def escape_snapshot_name(name):
    # square brackets cause a (de)serialization issue for TF checkpoints
    return name.replace("[", "(").replace("]", ")")


def fix_tf_gpu_memory_allocation(tf_sess_config):
    # TensorFlow 1.4.0 bug prevents setting GPU memory options for sessions
    # after listing available devices (device_lib.list_local_devices()).
    # A workaround is to create and destroy a temporary well-configured session
    # before listing available devices
    # ref: https://github.com/tensorflow/tensorflow/issues/9374
    # ref: https://github.com/tensorflow/tensorflow/issues/8021

    # unfortunately, this fix may cause another bug on windows (the issue
    # occurs when multiple sessions with a specific
    # per_process_gpu_memory_fraction are started
    if os.name != "nt":
        with tf.Session(config=tf_sess_config):  # bug-fix
            pass


def generator_summaries(g_out, experiment):
    raw_stats_summaries = stats_summaries(g_out, "raw-")
    raw_images_summaries = tf.summary.image(
        "raw-samples", g_out, experiment.eval_samples)
    sigmoid_images_summaries = tf.summary.image(
        "norm-samples", tf.sigmoid(g_out), experiment.eval_samples)
    images_summaries = tf.summary.merge([raw_images_summaries,
                                         sigmoid_images_summaries])
    return raw_stats_summaries, images_summaries


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def get_config_proto_attributes(config_proto):
    # return public attributes of default ConfigProto object
    return [i for i in getmembers(config_proto)
            if not i[0].startswith('_') and not ismethod(i[1])
            and not isbuiltin(i[1])]


def leaky_relu(x, leakiness=0.01):
    # porting of Lasagne's Leaky RELU activation function
    # http://lasagne.readthedocs.io/en/stable/modules/nonlinearities.html#
    # lasagne.nonlinearities.LeakyRectify
    # according to
    # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs
    leakiness = 0.01 if not leakiness or leakiness < 0 else leakiness
    return tf.maximum(x, leakiness * x)


def pad_left(tensor):
    return tf.expand_dims(tensor, 0)


def stats_summaries(var, prefix=""):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    mean_summ = tf.summary.scalar(prefix + "mean", mean)
    stddev_summ = tf.summary.scalar(prefix + "stddev", stddev)
    max_summ = tf.summary.scalar(prefix + "max", tf.reduce_max(var))
    min_summ = tf.summary.scalar(prefix + "min", tf.reduce_min(var))
    histogram_summ = tf.summary.histogram(prefix + "histogram", var)
    return tf.summary.merge(
        [mean_summ, stddev_summ, max_summ, min_summ, histogram_summ])


def train_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

import tensorboard
import tensorboard.backend.application as tb_conf
import tensorflow as tf
import utils_tf

from binomial_trainer import BinomialTrainer
from experiment import load_experiment_config
from utils_can import init_environment

# requirements.txt file was automatically generated with pipreqs
# $ pipreqs --force src/
# ref: https://github.com/bndr/pipreqshttps://github.com/bndr/pipreqs
# be careful with tensorflow/tensorflow-gpu versions!


def main():
    experiment_path = args.input
    experiment = load_experiment_config(experiment_path)
    logger = experiment.logger
    logger.info("Running experiment: {}".format(args.input))

    tf_sess_config = tf.ConfigProto()
    tf_sess_config.gpu_options.per_process_gpu_memory_fraction = args.fraction

    utils_tf.fix_tf_gpu_memory_allocation(tf_sess_config)

    logger.info("TensorFlow path: {}".format(tf.__file__))
    logger.info("TensorFlow version: {}".format(tf.__version__))
    logger.info("Available GPUs: {}\n".format(str(
        utils_tf.get_available_gpus())))

    logger.debug("TensorFlow ConfigProto configuration:")
    for a in utils_tf.get_config_proto_attributes(tf_sess_config):
        logger.debug(a)

    try:
        logger.info("TensorBoard path: {}".format(tensorboard.__file__))
        logger.info("TensorBoard conf path: {}".format(tb_conf.__file__))
    except AttributeError:  # elder tensorboard packages lack those attributes
        pass
    tb_default_conf = tb_conf.DEFAULT_TENSOR_SIZE_GUIDANCE
    logger.info("TensorBoard conf: {}".format(tb_default_conf))
    tb_params = [tb_conf.scalar_metadata.PLUGIN_NAME,
                 tb_conf.image_metadata.PLUGIN_NAME,
                 tb_conf.histogram_metadata.PLUGIN_NAME]
    msg = "TensorBoard parameter '{}' is {}. Set it to 0 to see all summaries."
    for param in filter(lambda p: tb_default_conf[p] != 0, tb_params):
        logger.warn(msg.format(param, tb_default_conf[param]))

    init_environment(experiment)

    logger.info("Loading dataset...")
    training, test, validation = experiment.get_dataset()
    msg = "Samples: {} training, {} test, {} validation\n"
    logger.info(msg.format(len(training), len(test), len(validation)))

    logger.info("Starting binomial trainer...")
    BinomialTrainer(experiment, tf_sess_config, training, test, validation)
    logger.info("Done!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the experiment .json")
    parser.add_argument("-f", "--fraction", type=float, default=0.9,
                        help="TensorFlow per-process GPU memory fraction")

    args = parser.parse_args()
    main()

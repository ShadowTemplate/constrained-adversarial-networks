import numpy as np
import random
import tensorflow as tf

from datetime import timedelta
from os import makedirs
from shutil import make_archive
from utils import min_max

LOG_TRAINING_EVERY = 10  # batches number
LOG_SUMMARIES_EVERY = 20  # batches number
PLOT_SAMPLES_EVERY = 1  # epochs number
STORE_CHECKPOINT_EVERY = 50  # epochs number
TEST_EVERY = 5  # epochs number
VALIDATE_EVERY = 999  # epochs number


def compress_folder(folder):
    make_archive(folder, "zip", folder)


def do_log_mb_progress(processed, bs, examples):
    # when statistics should be printed during training
    return (processed % (bs * LOG_TRAINING_EVERY)) == 0 \
           or processed + bs >= examples


def do_log_summaries(processed, bs, examples):
    # when tensorboard summaries should be collected during training
    return (processed % (bs * LOG_SUMMARIES_EVERY)) == 0 \
           or processed + bs >= examples


def do_plot_samples(epoch):
    # when evaluation noise should be used to create images during training
    return epoch % PLOT_SAMPLES_EVERY == 0


def do_store_checkpoint(epoch):
    # when checkpoints should be saved during training
    return epoch % STORE_CHECKPOINT_EVERY == 0


def do_test_network(epoch):
    # when ANN should be evaluated on test data
    return epoch % TEST_EVERY == 0


def do_validate_network(epoch):
    # when ANN should be evaluated on validation data
    return epoch % VALIDATE_EVERY == 0


def init_environment(experiment):
    set_seeds(experiment.ann_seed)  # make experiments reproducible

    for f in [experiment.output_folder, experiment.checkpoint_folder,
              experiment.pickle_folder, experiment.tensorboard_root]:
        makedirs(f, exist_ok=True)


def log_epoch_progress(logger, start_idx, bs, total, mbs_duration):
    processed_examples = start_idx + bs
    msg = "Processed {}/{} samples: {:.2f}%. Epoch ETA: {}"
    if do_log_mb_progress(processed_examples, bs, total):
        missing_mbs = (total - processed_examples) // bs
        mb_mean_time = np.mean(mbs_duration)
        epoch_ETA = timedelta(seconds=mb_mean_time * missing_mbs)
        logger.info(msg.format(
            processed_examples, total, processed_examples * 100 / total,
            str(epoch_ETA)))


def log_losses_info(logger, fn_name, epoch, losses, norm_factor):
    mb_num = len(losses)
    loss_mb_min, loss_mb_max = min_max(losses)
    loss_mb_min /= norm_factor
    loss_mb_max /= norm_factor
    avg_loss = sum(losses) / (mb_num * norm_factor)
    msg = "Epoch: {}, {} on batches: avg {}; min {}; max {}."
    logger.info(msg.format(epoch, fn_name, avg_loss, loss_mb_min, loss_mb_max))
    return avg_loss


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

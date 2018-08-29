import glob
import json
import numpy as np
import os
import random
import scipy.misc
import scipy.ndimage

from hashlib import md5
from inspect import getmembers, ismethod
from itertools import islice
from logging import DEBUG
from shutil import unpack_archive
from utils import get_logger, pickle_binary_file, unpickle_binary_file
from utils_tf import escape_snapshot_name as replace_brackets

if os.name != "nt":  # fuel seems to have some compatibility issues on Windows
    from fuel.datasets.hdf5 import H5PYDataset


class Experiment:
    def __init__(self, experiment_file, conf):
        # mandatory parameters: conf["PARAM"]
        # optional parameters: conf.get("PARAM")

        self.dataset_type = conf["DATASET_TYPE"]
        self.img_width = conf["IMG_WIDTH"]
        self.img_height = conf["IMG_HEIGHT"]

        if self.dataset_type == "polygons":
            self.dataset_seed = conf["DATASET_SEED"]
            self.dataset_size = conf["DATASET_SIZE"]
            self.polygons_number = conf["POLYGONS_NUMBER"]
            self.polygons_prob = conf["POLYGONS_PROB"]
            self.area = conf["AREA"]
            self.dataset_splits = tuple(
                conf["TRAINING_TEST_VALIDATION_SPLITS"])
            self.suffix_id = conf["SUFFIX_ID"]

            self.dataset_name = "{}k_{}x{}_pol{}_area{}_S{}{}".format(
                int(self.dataset_size // 1000), self.img_width,
                self.img_height, replace_brackets(str(self.polygons_prob)),
                self.area, self.dataset_seed, self.suffix_id).replace(" ", "_")
        else:  # "custom" type
            self.dataset_name = conf["DATASET_NAME"]
            self.dataset_loader_fn = conf["LOADER_FUNCTION"]
            self.loader_path = conf["LOADER_PATH"]

        self.ann_seed = conf["ANN_SEED"]
        self.generator = conf["GENERATOR"]
        self.discriminator = conf["DISCRIMINATOR"]
        self.generator_loss = conf["GENERATOR_LOSS"]
        self.discriminator_loss = conf["DISCRIMINATOR_LOSS"]

        self.batch_size = conf["BATCH_SIZE"]
        self.num_samples = conf["NUM_SAMPLES"]
        self.learning_rate = conf["LEARNING_RATE"]
        self.num_iter_gen = conf["NUM_ITER_GENERATOR"]
        self.num_iter_discr = conf["NUM_ITER_DISCRIMINATOR"]
        self.leak = conf["LEAKINESS"]
        self.z_dim = conf["z_dim"]
        self.h_dim = conf["h_dim"]

        self.learning_epochs = conf.get("LEARNING_EPOCHS")

        self.eval_samples = conf["EVAL_SAMPLES"]
        self.eval_noise_seed = conf.get("EVAL_NOISE_SEED")

        self.constraints = conf["CONSTRAINTS"]
        self.constraints_fn_single_batch = conf["CONSTRAINTS_FN_SINGLE_BATCH"]
        self.constraints_fn_multi_batch = conf["CONSTRAINTS_FN_MULTI_BATCH"]
        self.constrained_training = conf["CONSTRAINED_TRAINING"]
        self.constraints_epoch = conf.get("CONSTRAINTS_FROM_EPOCH")

        self.X_dim = [self.img_width, self.img_height, 1]

        if self.dataset_type == "custom":
            name = "{}_S{}_G{}_D{}_mb{}_ep{}_lr{}_zdim{}_hdim{}"
            self.name = name.format(
                experiment_file, self.ann_seed, self.generator,
                self.discriminator, self.batch_size, self.learning_epochs,
                self.learning_rate, self.z_dim, self.h_dim)
        else:  # "polygons" types
            name = "{}_ttv{}_{}_{}_S{}_G{}_D{}_mb{}_ep{}_lr{}_zdim{}_hdim{}"
            self.name = name.format(
                experiment_file, self.dataset_splits[0],
                self.dataset_splits[1], self.dataset_splits[2], self.ann_seed,
                self.generator, self.discriminator, self.batch_size,
                self.learning_epochs, self.learning_rate, self.z_dim,
                self.h_dim)

        self.datasets_folder = "out/datasets/"
        self.dataset_folder = self.datasets_folder + self.dataset_name + "/"
        folder_id = experiment_file + "/"
        self.output_folder = "out/images/" + folder_id
        self.checkpoint_folder = "out/model_checkpoints/" + folder_id
        self.checkpoint_file = "model"
        self.pickle_folder = self.datasets_folder + self.dataset_name + \
                             "_pickle/"
        self.tensorboard_root = "out/tensorboard/"
        self.tb_experiment_folder = self.tensorboard_root + folder_id
        self.py_random_state = "py_random_state.pkl"
        self.np_random_state = "np_random_state.pkl"
        self.validation_stats_window = 5
        self.validation_stats = "validation_stats.dill"
        self.test_stats = "test_stats.dill"
        self.constrained_flag = "constrained_flag.txt"
        # since Python's str.__hash__ is non-deterministic across different
        # runs it is necessary to use hashlib
        constraints_hash = md5("".join(self.constraints).encode()).hexdigest()
        self.constraints_training_cache_file = \
            self.pickle_folder + "cache_constraints_training_{}_chash_{}" \
                                 ".pkl".format(self.ann_seed, constraints_hash)
        self.constraints_test_cache_file = \
            self.pickle_folder + "cache_constraints_test_{}_chash_{}" \
                                 ".pkl".format(self.ann_seed, constraints_hash)
        self.constraints_validation_cache_file = \
            self.pickle_folder + "cache_constraints_validation_{}_chash_{}" \
                                 ".pkl".format(self.ann_seed, constraints_hash)

        self.log_folder = "out/log/"
        os.makedirs(self.log_folder, exist_ok=True)
        self.logger = get_logger(
            __name__, DEBUG, DEBUG, self.log_folder + self.name + ".log")

    def __repr__(self):
        return str(self.__dict__)

    def plot_images(self, samples, epoch):
        # samples shape: (eval_samples, width, height, 1)
        image_path = self.output_folder + "{}_{}.png"
        for j, image in enumerate(samples):
            image = image.reshape((self.img_width, self.img_height))
            scipy.misc.imsave(image_path.format(
                str(epoch).zfill(6), str(j).zfill(2)), image)

    def get_dataset(self):
        # train/test/validation splits for specific seed
        files = [f.format(self.ann_seed) for f in [
            "training_{}.pkl", "test_{}.pkl", "validation_{}.pkl"]]
        if all(map(lambda f: os.path.exists(self.pickle_folder + f), files)):
            self.logger.info("Pickle dataset files found for seed {}. Going "
                             "to load them...".format(self.ann_seed))
            return self._load_dataset_from_files(files)

        self.logger.info("Pickle dataset files not found for seed {}. "
                         "Creating dataset...".format(self.ann_seed))
        if self.dataset_type == "custom":
            fns = dict(getmembers(self, predicate=ismethod))
            training, test, validation = fns[self.dataset_loader_fn]()
        else:  # "polygons" types
            training, test, validation = self._create_from_images()
        self.logger.info("Dataset created. Saving pickle dataset files...")
        self._save_dataset_to_files(files, training, test, validation)
        return training, test, validation

    def _load_dataset_from_files(self, files):
        return tuple(unpickle_binary_file(self.pickle_folder + f)
                     for f in files)

    def _save_dataset_to_files(self, files, training, test, validation):
        for split in zip(files, [training, test, validation]):
            pickle_binary_file(self.pickle_folder + split[0], split[1])

    def _create_from_images(self):
        # extract dataset archive if necessary
        if not os.path.exists(self.dataset_folder):
            archive = self.dataset_name + ".zip"
            archive_path = self.datasets_folder + archive
            msg = "Extracting dataset archive {}..."
            self.logger.info(msg.format(archive_path))
            unpack_archive(archive_path, self.dataset_folder, 'zip')

        msg = "Creating dataset from folder {}..."
        self.logger.info(msg.format(self.dataset_folder))
        files_names = glob.glob(glob.escape(self.dataset_folder) + "*.png")
        random.seed(self.ann_seed)
        random.shuffle(files_names)  # guarantees consistency for ttv splits
        images_data = self._read_binomial_dataset(files_names)
        it = iter(images_data)
        training, test, validation = (list(islice(it, 0, i))
                                      for i in self.dataset_splits)

        return self._reshape_samples(training) if len(training) > 0 else [], \
               self._reshape_samples(test) if len(test) > 0 else [], \
               self._reshape_samples(validation) if len(validation) > 0 else []

    def _read_binomial_dataset(self, files_names):
        images_data = []
        for image in files_names:
            # read image as greyscale
            image_data = scipy.ndimage.imread(image, flatten=True)
            image_data[image_data == 255] = 1  # binarize (255 -> 1)
            images_data.append(image_data)
        return images_data

    def _binarized_mnist_loader(self):
        examples, dataset_splits = [], []
        for split in ["train", "test", "valid"]:
            dataset = H5PYDataset(self.loader_path, which_sets=(split,))
            data_stream = dataset.get_example_stream()
            data = list(data_stream.get_epoch_iterator())
            examples += data
            dataset_splits.append(len(data))

        random.seed(self.ann_seed)
        random.shuffle(examples)  # guarantees consistency for ttv splits
        it = iter(examples)
        return (self._reshape_samples(list(islice(it, 0, i)))
                for i in dataset_splits)

    def _reshape_samples(self, samples):
        return np.reshape(np.stack(samples), newshape=(
            -1, self.img_width, self.img_height, 1))


def load_experiment_config(experiment_path):
    with open(experiment_path, "r") as conf_f:
        filename = experiment_path.split(os.sep)[-1]
        experiment_json = json.load(conf_f)
        print("Deserialized JSON: {}".format(
            json.dumps(experiment_json, indent=4)))
        return Experiment(filename.split(".")[0], experiment_json)

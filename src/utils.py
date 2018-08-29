import dill
import logging
import numpy as np
import os
import pickle
import random
import sys

from inspect import getmembers, isfunction
from numba.targets.registry import CPUDispatcher as numba_compiled_fn
from xxhash import xxh32


class HashableArray:
    # provide a fast hashable wrapper for numpy arrays

    def __init__(self, array):
        self.array = array

    def __hash__(self):
        # xxhash seems to be the fastest lib to hash numpy arrays.
        # an improvement in this method may result in an overall considerable
        # improvement, so it may be worth it to dig in deeper
        return xxh32(self.array.view(self.array.dtype)).intdigest()

    def __getitem__(self, index):
        return self.array[index]

    def __repr__(self):
        return repr(self.array)

    def __eq__(self, other):
        return np.array_equal(self.array, other.array)


def dill_binary_file(file_name, item):
    with open(file_name, mode="wb") as output_f:
        dill.dump(item, output_f)


def first_arg_null_safe(f):
    # call f only if its first argument is not None.
    return lambda *fargs, **fkwargs: f(*fargs, **fkwargs) if fargs[0] else None


def get_logger(module_name, stdout_level, file_level, log_file):
    msg_pattern = '[%(asctime)s] {%(filename)s:%(funcName)s:%(lineno)d} ' \
                  '%(levelname)s: %(message)s'
    date_pattern = '%d/%m/%Y %H:%M:%S'  # keep consistent with plots.py

    # set up logging to file
    logging.basicConfig(filename=log_file, level=file_level,
                        format=msg_pattern, datefmt=date_pattern)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(stdout_level)
    console.setFormatter(logging.Formatter(msg_pattern, date_pattern))
    logging.getLogger('').addHandler(console)
    # if not hasattr(get_logger, '_console_handler_set'):
    #     get_logger._console_handler_set = True
    #     # set up logging to console
    #     console = logging.StreamHandler()
    #     console.setLevel(stdout_level)
    #     console.setFormatter(logging.Formatter(msg_pattern, date_pattern))
    #     logging.getLogger('').addHandler(console)
    return logging.getLogger(module_name)


def get_module_functions(module_name):
    def is_numba_compiled_fn(o):
        return type(o) == numba_compiled_fn

    def is_module_function(o):
        return is_numba_compiled_fn(o) or (
                isfunction(o) and o.__module__ == module_name)

    return dict(getmembers(sys.modules[module_name], is_module_function))


def load_from_text_file(file_name):
    with open(file_name, mode="r") as input_f:
        return input_f.readline()


def load_random_states(experiment, epoch_folder):
    random.setstate(unpickle_binary_file(
        epoch_folder + experiment.py_random_state))
    np.random.set_state(unpickle_binary_file(
        epoch_folder + experiment.np_random_state))


def min_max(items):
    curr_min = curr_max = items[0]
    for i in items:
        if i < curr_min:
            curr_min = i
        if i > curr_max:
            curr_max = i
    return curr_min, curr_max


def pickle_binary_file(file_name, item):
    with open(file_name, mode="wb") as output_f:
        pickle.dump(item, output_f)


def remove_folder(folder):
    # recursively remove folders
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def save_as_text_file(file_name, content):
    with open(file_name, mode="w") as output_f:
        output_f.write(str(content))


def save_random_states(experiment, epoch_folder):
    pickle_binary_file(epoch_folder + experiment.py_random_state,
                       random.getstate())
    pickle_binary_file(epoch_folder + experiment.np_random_state,
                       np.random.get_state())


def undill_binary_file(file_name):
    with open(file_name, mode="rb") as input_f:
        return dill.load(input_f)


def unpickle_binary_file(file_name):
    with open(file_name, mode="rb") as input_f:
        return pickle.load(input_f)

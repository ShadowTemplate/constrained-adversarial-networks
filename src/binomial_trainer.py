import architectures
import constraints as constraints_module
import numpy as np
import tensorflow as tf
import utils
import utils_can
import utils_tf

from datetime import timedelta
from losses import get_losses
from os import makedirs, walk, path
from solvers import get_solvers
from stats_windows import TestWindow, ValidationWindow
from tensorflow.python.framework.errors_impl import (
    NotFoundError as CheckpointNotFoundError)
from time import time
from utils import first_arg_null_safe


class BinomialTrainer:
    def __init__(self, experiment, tf_session_config, training_data, test_data,
                 validation_data):
        # keep calm and take a deep breath: there a LOT of things to define
        # and initialize before running the training algorithm...

        self.logger = experiment.logger

        # input data
        self.experiment = experiment
        self.experiment.channels = 1
        self.training_data = training_data
        self.test_data = test_data
        self.validation_data = validation_data

        # training constants
        self.bs = experiment.batch_size
        self.n_samples = experiment.num_samples
        self.num_examples = len(training_data)
        self.num_examples_test = len(test_data)
        self.num_examples_validation = len(validation_data)
        self.num_constraints = len(experiment.constraints)
        self.batches_indices = np.arange(self.num_examples)
        self.constraints_epoch = experiment.constraints_epoch

        # TF variables to be serialized and their corresponding update ops
        self.curr_epoch_var = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.increment_curr_epoch_op = tf.assign_add(self.curr_epoch_var, 1)

        # placeholders for training
        self.R_shape = [self.n_samples, self.bs] + experiment.X_dim
        self.X = tf.placeholder(
            tf.float32, [None] + experiment.X_dim, "X_placeholder")
        self.C_real = tf.placeholder(
            tf.float32, [None, self.num_constraints], "C_real_placeholder")
        self.z = tf.placeholder(
            tf.float32, [None, experiment.z_dim], "z_placeholder")
        self.R = tf.placeholder(tf.float32, self.R_shape, "R_placeholder")
        self.C_fake = tf.placeholder(
            tf.float32, [None, self.num_constraints], "C_fake_placeholder")

        # placeholders for evaluation
        self.g_eval_logit = tf.placeholder(
            tf.float32, [experiment.eval_samples] + experiment.X_dim,
            "g_eval_logit_placeholder")
        self.eval_g_op = tf.sigmoid(self.g_eval_logit)

        # statistics for validation/test
        self.validation_window = ValidationWindow(
            experiment.validation_stats_window, experiment.constraints,
            self.logger)
        self.test_window = TestWindow(experiment.constraints, self.logger)

        # possible constraints
        constraints_fns = utils.get_module_functions(
            constraints_module.__name__)
        self.constraints_fn_single_batch = constraints_fns[
            self.experiment.constraints_fn_single_batch]
        self.constraints_fn_multi_batch = constraints_fns[
            self.experiment.constraints_fn_multi_batch]
        self.constraints_cache_training = None
        self.constraints_cache_test = None
        self.constraints_cache_validation = None

        self.use_constraints = False

        # ANN architecture, loss functions, SGD algorithms
        with tf.variable_scope("can"):
            discriminator_fn, generator_fn = self._build_adversarial_network()

            # generate probabilities from random input noise
            G_kwargs = {"experiment": experiment, "z": self.z}
            self.G_output_logit, G_stats_summaries, self.G_images_summaries = \
                generator_fn(**G_kwargs)
            G_output = tf.sigmoid(self.G_output_logit)
            self.G_sample = tf.cast(
                self.R <= utils_tf.pad_left(G_output), tf.float32)

            D_real_kwargs = {"experiment": experiment, "X": self.X}
            # discriminator is used twice (real/fake data): must enable reuse
            D_fake_kwargs = {
                "experiment": experiment,
                "X": tf.reshape(self.G_sample, [-1] + experiment.X_dim),
                "reuse": True}

            if experiment.constrained_training:
                # discriminator parameters dictionary needs constraints info
                D_constraints_kwargs = {
                    "constraints_epoch": self.constraints_epoch,
                    "use_constraints": self.use_constraints}
                D_real_kwargs = {**D_real_kwargs, **D_constraints_kwargs,
                                 "constraints_features": self.C_real}
                D_fake_kwargs = {**D_fake_kwargs, **D_constraints_kwargs,
                                 "constraints_features": self.C_fake}

            D_real, D_real_stats_summaries = discriminator_fn(**D_real_kwargs)
            D_fake, D_fake_stats_summaries = discriminator_fn(**D_fake_kwargs)

            self.D_loss, D_loss_summaries, self.G_loss, G_loss_summaries = \
                get_losses(experiment, self.G_sample, D_real, D_fake,
                           self.G_output_logit, self.n_samples, self.bs)

            self.D_summaries = tf.summary.merge(
                [D_real_stats_summaries, D_fake_stats_summaries,
                 D_loss_summaries])
            self.G_summaries = tf.summary.merge(
                [G_stats_summaries, G_loss_summaries])

            # parameters to update with stochastic gradient descent (SGD)
            theta_G = utils_tf.train_vars("can/generator")
            theta_D = utils_tf.train_vars("can/discriminator")

            # SGD algorithms to update parameters according to loss functions
            self.D_solver, self.G_solver = get_solvers(
                experiment, self.D_loss, theta_D, self.G_loss, theta_G)

        self.saver = tf.train.Saver(max_to_keep=None)

        # define sampling functions
        self.z_mb_fn = lambda n: np.random.rand(n, experiment.z_dim)
        random_noise_fn = tf.random_uniform(self.R_shape)
        self.R_mb_fn = lambda: random_noise_fn.eval()

        with tf.Session(config=tf_session_config) as session:
            self.session = session
            self.session.run(tf.global_variables_initializer())
            # load preexisting session if possible
            self._maybe_load_session()

            # define where to collect summaries
            self.tb_writer = tf.summary.FileWriter(
                experiment.tb_experiment_folder, self.session.graph)
            # make tb_writer None-safe for convenience (we log summaries only
            # periodically during training: this tb_writer will properly
            # handle the absence of summaries)
            self.tb_writer.add_summary = first_arg_null_safe(
                self.tb_writer.add_summary)

            # proxy tensor of constraints that can be used for real mini-batch
            # (e.g: whenever epoch < constraints_epoch)
            self.proxy_constraints_real = tf.zeros([
                self.bs, self.num_constraints], tf.float32).eval()
            # proxy tensor of constraints that can be used for fake mini-batch
            # (e.g: whenever epoch < constraints_epoch)
            self.proxy_constraints_fake = tf.zeros([
                self.n_samples * self.bs, self.num_constraints],
                tf.float32).eval()

            # random noise to evaluate generator. It is kept fixed to easily
            # observe progresses during training in tensorboard
            # if a seed is provided, it is used (handy to have the same noise
            # for the same experiment).
            if experiment.eval_noise_seed:
                msg = "Sampling eval noise with custom seed: {}"
                self.logger.info(msg.format(experiment.eval_noise_seed))
                old_random_state = np.random.get_state()
                np.random.seed(experiment.eval_noise_seed)
                self.eval_z = self.z_mb_fn(experiment.eval_samples)
                np.random.set_state(old_random_state)
            else:
                self.eval_z = self.z_mb_fn(experiment.eval_samples)

            self.curr_epoch = self.curr_epoch_var.eval(self.session)
            self.logger.info("Starting from epoch: {}".format(self.curr_epoch))

            # make the graph read-only to prevent memory leaks
            tf.get_default_graph().finalize()

            if self.curr_epoch != experiment.learning_epochs:
                self._train()  # FINALLY TRAINING!
                self.logger.info("Training completed.")

            self.logger.info("Performing final test with fixed seed...")
            self._test(self.experiment.ann_seed)
            self.test_window.print_stats()

    def _build_adversarial_network(self):
        # get the discriminator and generator that will be used for the
        # adversarial training. The types of the discriminator and generator
        # will determine if it is a gan or a can.
        fns = utils.get_module_functions(architectures.__name__)
        discriminator_fn = fns[self.experiment.discriminator]
        generator_fn = fns[self.experiment.generator]
        return discriminator_fn, generator_fn

    def _compute_constraints_fake(self, noise_tensors):
        # generate images from noise
        R_mb = self.G_sample.eval(noise_tensors)

        constraints_values = self.constraints_fn_multi_batch(
            R_mb, self.n_samples, self.bs, self.num_constraints,
            self.experiment.area, self.experiment.polygons_number,
            self.experiment.img_width, self.experiment.img_height)
        return constraints_values.reshape(-1, constraints_values.shape[2])

    def _compute_constraints_real(self, data):
        self.logger.info("Computing constraints on real data...")
        if len(data) == 0:
            return []
        return self.constraints_fn_single_batch(
            data, self.num_constraints, self.experiment.area,
            self.experiment.polygons_number, self.experiment.img_width,
            self.experiment.img_height)

    def _discriminator_step(self, ops, real_tensors, compute_fake_constraints):
        noise_tensors, fake_constraints_tensor = self._prepare_tensors(
            compute_fake_constraints)
        feed_tensors = {**real_tensors, **noise_tensors,
                        **fake_constraints_tensor}
        # run SGD algorithm
        D_loss_curr, _, summaries = self.session.run(ops, feed_tensors)
        self.tb_writer.add_summary(summaries, self.curr_epoch)
        return D_loss_curr, fake_constraints_tensor

    def _evaluate_constraints_compliance(self, fake_constraints_errors):
        num_stats_samples = len(fake_constraints_errors)
        averages, percentages = [], []
        for j, fn_name in enumerate(self.experiment.constraints):
            # filter errors for the current constraint function
            errors = [err_tuple[j] for err_tuple in fake_constraints_errors]
            min_error, max_error = utils.min_max(errors)
            avg_error = np.average(errors)
            averages.append(avg_error)
            self.logger.info("E[{}]: {}".format(fn_name, avg_error))
            self.logger.info("Var[{}]: {}".format(fn_name, np.var(errors)))
            self.logger.info("min({}): {}".format(fn_name, min_error))
            self.logger.info("max({}): {}".format(fn_name, max_error))
            zero_error = errors.count(0)
            perfect_percentage = 100 * zero_error / num_stats_samples
            percentages.append(perfect_percentage)
            self.logger.info("perfect({}): {}/{} ({}%)".format(
                fn_name, zero_error, num_stats_samples, perfect_percentage))

        if self.num_constraints > 0:
            all_zero = len([t for t in fake_constraints_errors if sum(t) == 0])
            perfect_percentage = 100 * all_zero / num_stats_samples
            percentages.append(perfect_percentage)
            self.logger.info("perfect(ALL): {}/{} ({}%)".format(
                all_zero, num_stats_samples, perfect_percentage))
        return averages, percentages

    def _generate_evaluation_images(self):
        # save images to tensorboard starting from constant eval_z noise
        G_sample_eval_logit, images_summaries = self.session.run(
            [self.G_output_logit, self.G_images_summaries],
            feed_dict={self.z: self.eval_z})
        # G_sample_eval_logit shape: (eval_samples, width, height, 1)
        self.tb_writer.add_summary(images_summaries, self.curr_epoch)

        # reuse the computed logit to build images to be also saved on file
        G_samples = self.eval_g_op.eval(
            feed_dict={self.g_eval_logit: G_sample_eval_logit})
        self.experiment.plot_images(G_samples, self.curr_epoch + 1)
        msg = "Generated samples saved in folder: {}"
        self.logger.info(msg.format(self.experiment.output_folder))

    def _generator_step(self, ops, compute_fake_constraints):
        noise_tensors, fake_constraints_tensor = self._prepare_tensors(
            compute_fake_constraints)
        feed_tensors = {**noise_tensors, **fake_constraints_tensor}
        # run SGD algorithm
        G_loss_curr, _, summaries = self.session.run(ops, feed_tensors)
        self.tb_writer.add_summary(summaries, self.curr_epoch)
        return G_loss_curr, fake_constraints_tensor

    def _get_constraints_real(self, data, cache_file):
        if path.exists(cache_file):
            self.logger.info("Retrieving constraints values from cache: {}".format(
                cache_file))
            constraints_values = utils.unpickle_binary_file(cache_file)
        else:
            self.logger.info(
                "Unable to retrieve constraints values from cache: {}. Going "
                "to compute them...".format(cache_file))
            constraints_values = self._compute_constraints_real(data)
            self.logger.info("Storing constraints values in {}".format(
                cache_file))
            utils.pickle_binary_file(cache_file, constraints_values)
        return constraints_values

    def _get_fake_constraints_tensor(self, noise_tensors, compute):
        # return a dictionary containing, if necessary, the constraints tensor
        # for a fake mini-batch
        if not compute:
            return {self.C_fake: self.proxy_constraints_fake}

        # compute constraints features of fake mini-batch
        return {self.C_fake: self._compute_constraints_fake(noise_tensors)}

    def _get_real_constraints_tensor(self, batch_indices,
                                     constraints_cache, compute):
        # return a dictionary containing, if necessary, the constraints tensor
        # for a data mini-batch from training, validation or test
        if not compute:
            return {self.C_real: self.proxy_constraints_real}

        return {self.C_real: constraints_cache[batch_indices]}

    def _maybe_load_session(self):
        try:
            # list available checkpoints and pick the one from the latest epoch
            _, dirs, _ = next(walk(self.experiment.checkpoint_folder))
            dirs.sort()
            last_epoch = self.experiment.checkpoint_folder + dirs[-1] + "/"
            checkpoint_model = last_epoch + self.experiment.checkpoint_file
            self.saver.restore(self.session, checkpoint_model)
            utils.load_random_states(self.experiment, last_epoch)
            self.validation_window = utils.undill_binary_file(
                last_epoch + self.experiment.validation_stats)
            self.test_window = utils.undill_binary_file(
                last_epoch + self.experiment.test_stats)
            # windows' loggers must be reset
            self.validation_window.logger = self.logger
            self.test_window.logger = self.logger
            self.use_constraints = eval(utils.load_from_text_file(
                last_epoch + self.experiment.constrained_flag))
            msg = "Successfully loaded checkpoint file: {}\n"
            self.logger.info(msg.format(checkpoint_model))
        except (CheckpointNotFoundError, IndexError) as ex:
            if type(ex) == IndexError:
                self.logger.info("Unable to find any checkpoint")
            else:
                self.logger.exception("Unable to load checkpoint correctly")
            msg = "Starting from scratch. Unable to load checkpoint from: {}\n"
            self.logger.info(msg.format(self.experiment.checkpoint_folder))
            # tensorboard experiment folder is cleaned iff there is no
            # checkpoint, because it is desirable to keep events from previous
            # runs to display more informative statistics
            utils.remove_folder(self.experiment.tb_experiment_folder)
            makedirs(self.experiment.tb_experiment_folder, exist_ok=True)

    def _prepare_tensors(self, compute_fake_constraints):
        # sample fake examples
        fake_examples_continuous = self.R_mb_fn()
        # sample noise for discriminator input layer
        z_mb = self.z_mb_fn(self.bs)
        noise_tensors = {self.z: z_mb, self.R: fake_examples_continuous}
        # compute constraints for fake data if necessary
        fake_constraints_tensor = self._get_fake_constraints_tensor(
            noise_tensors, compute_fake_constraints)
        return noise_tensors, fake_constraints_tensor

    def _run_test_epoch(self, D_ops):
        # this function is similar to _run_train_epoch, but only runs the
        # discriminator on the test set without updating its weights
        D_losses = []
        mbs_duration = []

        # do not collect summaries for this epoch
        # define a TF "no-op" that can be handled by the None-safe tb_writer
        D_summaries_op = [[]]
        ops = D_ops + D_summaries_op
        fake_constraints_errors = []

        self.logger.debug("Using constraints features for real mini-batch")
        self.logger.debug("Computing constraints features for fake mini-batch")

        for start_idx in range(0, self.num_examples_test - self.bs + 1,
                               self.bs):
            mb_start_time = time()
            batches_indices = np.arange(self.num_examples_test)
            batch_indices = batches_indices[start_idx:start_idx + self.bs]
            X_mb = self.test_data[batch_indices]
            # always compute constraints at test time
            constraints_tensor = self._get_real_constraints_tensor(
                batch_indices, self.constraints_cache_test, True)
            real_tensors = {self.X: X_mb, **constraints_tensor}

            # each batch can be used to train the discriminator and the
            # generator a different number of times (num_iter_discr vs
            # num_iter_gen)
            for i in range(self.experiment.num_iter_discr):
                # always compute constraints at test time
                D_loss_mb, fake_constraints_tensor = self._discriminator_step(
                    ops, real_tensors, True)
                fake_constraints_errors += [
                    tuple(err) for err in fake_constraints_tensor[self.C_fake]]
                D_losses.append(D_loss_mb)

            # log running time information if needed
            mbs_duration.append(time() - mb_start_time)
            utils_can.log_epoch_progress(
                self.logger, start_idx, self.bs, self.num_examples_test,
                mbs_duration)

        return D_losses, fake_constraints_errors

    def _run_train_epoch(self, D_ops, G_ops):
        D_losses = []
        G_losses = []
        mbs_duration = []

        self.logger.info(
            "Constrained training: {}".format(self.use_constraints))

        if not self.use_constraints:
            msg = "Using proxy constraints features for real mini-batch"
            self.logger.debug(msg)
            msg = "Using proxy constraints features for fake mini-batch"
            self.logger.debug(msg)
        else:
            self.logger.debug("Using constraints features for real mini-batch")
            self.logger.debug(
                "Computing constraints features for fake mini-batch")

        for start_idx in range(0, self.num_examples - self.bs + 1, self.bs):
            mb_start_time = time()
            batch_indices = self.batches_indices[start_idx:start_idx + self.bs]
            X_mb = self.training_data[batch_indices]
            # compute constraints only if architecture supports them
            constraints_tensor = self._get_real_constraints_tensor(
                batch_indices, self.constraints_cache_training,
                self.use_constraints)
            real_tensors = {self.X: X_mb, **constraints_tensor}

            # collect summaries periodically, not after every batch
            if utils_can.do_log_summaries(
                    start_idx, self.bs, self.num_examples):
                D_summaries_op = [self.D_summaries]
                G_summaries_op = [self.G_summaries]
            else:
                # define a TF "no-op" that can be handled by the
                # None-safe tb_writer
                D_summaries_op = G_summaries_op = [[]]

            # each batch can be used to train the discriminator and the
            # generator a different number of times (num_iter_discr vs
            # num_iter_gen)
            ops = D_ops + D_summaries_op
            for i in range(self.experiment.num_iter_discr):
                D_loss_mb, _ = self._discriminator_step(
                    ops, real_tensors, self.use_constraints)
                D_losses.append(D_loss_mb)

            ops = G_ops + G_summaries_op
            for i in range(self.experiment.num_iter_gen):
                G_loss_mb, _ = self._generator_step(ops, self.use_constraints)
                G_losses.append(G_loss_mb)

            # log running time information if needed
            mbs_duration.append(time() - mb_start_time)
            utils_can.log_epoch_progress(
                self.logger, start_idx, self.bs, self.num_examples, mbs_duration)

        return D_losses, G_losses

    def _run_validation_epoch(self, D_ops, G_ops):
        # this function is similar to _run_train_epoch, but runs the
        # discriminator and generator on the validation set without updating
        # their weights
        D_losses = []
        G_losses = []
        mbs_duration = []

        # do not collect summaries for this epoch
        # define a TF "no-op" that can be handled by the None-safe tb_writer
        D_summaries_op = G_summaries_op = [[]]
        fake_constraints_errors = []

        self.logger.debug("Using constraints features for real mini-batch")
        self.logger.debug("Computing constraints features for fake mini-batch")

        for start_idx in range(0, self.num_examples_validation - self.bs + 1,
                               self.bs):
            mb_start_time = time()
            batches_indices = np.arange(self.num_examples_validation)
            batch_indices = batches_indices[start_idx:start_idx + self.bs]
            X_mb = self.validation_data[batch_indices]
            # always compute constraints at validation time
            constraints_tensor = self._get_real_constraints_tensor(
                batch_indices, self.constraints_cache_validation, True)
            real_tensors = {self.X: X_mb, **constraints_tensor}

            # each batch can be used to train the discriminator and the
            # generator a different number of times (num_iter_discr vs
            # num_iter_gen)
            ops = D_ops + D_summaries_op
            for i in range(self.experiment.num_iter_discr):
                # always compute constraints at validation time
                D_loss_mb, fake_constraints_tensor = self._discriminator_step(
                    ops, real_tensors, True)
                fake_constraints_errors += [
                    tuple(err) for err in fake_constraints_tensor[self.C_fake]]
                D_losses.append(D_loss_mb)

            ops = G_ops + G_summaries_op
            for i in range(self.experiment.num_iter_gen):
                G_loss_mb, _ = self._generator_step(ops, self.use_constraints)
                G_losses.append(G_loss_mb)

            # log running time information if needed
            mbs_duration.append(time() - mb_start_time)
            utils_can.log_epoch_progress(
                self.logger, start_idx, self.bs, self.num_examples_validation,
                mbs_duration)

        return D_losses, G_losses, fake_constraints_errors

    def _store_session(self):
        if self.curr_epoch == 0:  # happens if you add constraints from epoch 0
            return  # no training has occurred: no need to store anything
        epoch_number = str(self.curr_epoch - 1).zfill(6)
        epoch_folder = self.experiment.checkpoint_folder + "epoch_" + \
                       epoch_number + "/"
        makedirs(epoch_folder, exist_ok=True)
        checkpoint_path = epoch_folder + self.experiment.checkpoint_file
        save_path = self.saver.save(self.session, checkpoint_path)
        utils.save_random_states(self.experiment, epoch_folder)
        validation_path = epoch_folder + self.experiment.validation_stats
        utils.dill_binary_file(validation_path, self.validation_window)
        test_path = epoch_folder + self.experiment.test_stats
        utils.dill_binary_file(test_path, self.test_window)
        flag_path = epoch_folder + self.experiment.constrained_flag
        utils.save_as_text_file(flag_path, self.use_constraints)
        self.logger.info("Model saved in file: " + save_path)

    def _test(self, seed=None):
        # this function is similar to _train, but only runs the discriminator
        # on the test set without updating its weights for one epoch

        # compute and cache constraints features for test examples
        if self.constraints_cache_test is None:
            self.constraints_cache_test = self._get_constraints_real(
                self.test_data, self.experiment.constraints_test_cache_file)

        if seed:
            utils_can.set_seeds(seed)

        # define operations to run
        D_ops = [self.D_loss, []]  # note: D_solver is not passed

        # only one epoch is run: no need to compute ETA
        self.logger.info("Starting TEST epoch")
        D_losses, fake_constraints_errors = self._run_test_epoch(D_ops)
        self.logger.info("Completed TEST epoch")

        avg_D_loss = utils_can.log_losses_info(
            self.logger, "D_loss", "TEST", D_losses,
            self.experiment.num_iter_discr)

        self.test_window.add_losses(self.curr_epoch, avg_D_loss)
        num_stats_samples = len(fake_constraints_errors)
        msg = "Computing stats on the {} items generated during test epoch..."
        self.logger.info(msg.format(num_stats_samples))

        averages, percentages = self._evaluate_constraints_compliance(
            fake_constraints_errors)
        self.test_window.add_constraints_stats(
            self.curr_epoch, averages, percentages)

    def _train(self):
        if self.experiment.constrained_training:
            # compute and cache constraints features for dataset examples
            self.constraints_cache_training = self._get_constraints_real(
                self.training_data,
                self.experiment.constraints_training_cache_file)
        self.constraints_cache_validation = self._get_constraints_real(
            self.validation_data,
            self.experiment.constraints_validation_cache_file)
        self.constraints_cache_test = self._get_constraints_real(
            self.test_data, self.experiment.constraints_test_cache_file)

        epochs_duration = []

        # define operations to run
        D_ops = [self.D_loss, self.D_solver]
        G_ops = [self.G_loss, self.G_solver]

        # ANN training
        while self.curr_epoch < self.experiment.learning_epochs:
            epoch_start_time = time()

            if self.constraints_epoch is not None and \
                    self.curr_epoch >= self.constraints_epoch:
                self.use_constraints = True

            np.random.shuffle(self.batches_indices)
            self.logger.info(
                "Starting TRAINING epoch {}".format(self.curr_epoch))
            D_losses, G_losses = self._run_train_epoch(D_ops, G_ops)
            self.logger.info(
                "Completed TRAINING epoch {}".format(self.curr_epoch))

            # log running time information
            epochs_duration.append(time() - epoch_start_time)
            epoch_mean_time = np.mean(epochs_duration)
            missing_epochs = \
                self.experiment.learning_epochs - self.curr_epoch - 1
            training_ETA = timedelta(seconds=epoch_mean_time * missing_epochs)

            utils_can.log_losses_info(self.logger, "D_loss", self.curr_epoch,
                                      D_losses, self.experiment.num_iter_discr)
            utils_can.log_losses_info(self.logger, "G_loss", self.curr_epoch,
                                      G_losses, self.experiment.num_iter_gen)
            self.logger.info("Training ETA: {}".format(str(training_ETA)))

            # save evaluation images to tensorboard and to file if necessary
            if utils_can.do_plot_samples(self.curr_epoch):
                self._generate_evaluation_images()

            self.curr_epoch = self.session.run(self.increment_curr_epoch_op)

            if utils_can.do_validate_network(self.curr_epoch):
                self._validate()

            if utils_can.do_test_network(self.curr_epoch):
                self._test()

            # save training checkpoint if necessary
            if utils_can.do_store_checkpoint(self.curr_epoch):
                self._store_session()

        self.logger.info("Reached the maximum epochs number: training stopped")

        self._store_session()  # always store final checkpoint
        msg = "Compressing {} folder..."
        for f in [self.experiment.output_folder,
                  self.experiment.checkpoint_folder,
                  self.experiment.tb_experiment_folder]:
            self.logger.debug(msg.format(f))
            utils_can.compress_folder(f)

    def _validate(self):
        # this function is similar to _train, but runs the discriminator and
        # the generator on the validation set without updating their weights
        # for one epoch

        # define operations to run
        D_ops = [self.D_loss, []]  # note: D_solver is not passed
        G_ops = [self.G_loss, []]  # note: G_solver is not passed

        # only one epoch is run: no need to compute ETA
        self.logger.info("Starting VALIDATION epoch")
        D_losses, G_losses, fake_constraints_errors = \
            self._run_validation_epoch(D_ops, G_ops)
        self.logger.info("Completed VALIDATION epoch")

        avg_D_loss = utils_can.log_losses_info(
            self.logger, "D_loss", "VALIDATION", D_losses,
            self.experiment.num_iter_discr)

        avg_G_loss = utils_can.log_losses_info(
            self.logger, "G_loss", "VALIDATION", G_losses,
            self.experiment.num_iter_gen)

        self.validation_window.add_losses(avg_D_loss, avg_G_loss)
        num_stats_samples = len(fake_constraints_errors)
        msg = "Computing stats on the {} items generated during validation " \
              "epoch..."
        self.logger.info(msg.format(num_stats_samples))
        averages, percentages = self._evaluate_constraints_compliance(
            fake_constraints_errors)
        self.validation_window.add_constraints_stats(
            self.curr_epoch, averages, percentages)

        try:
            # TODO remove this log
            weights = self.session.run(
                tf.get_default_graph().get_tensor_by_name(
                "can/discriminator/constrained_out/d_constraints_kernel:0"))
            self.logger.info(
                "Constraints weights: {}".format(weights.flatten()))
        except KeyError:
            pass

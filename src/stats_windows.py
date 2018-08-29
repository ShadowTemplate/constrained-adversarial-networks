import numpy as np

MIN_LOSS_STD = 0.05
MAX_EPOCHS_WITH_NO_IMPROVEMENTS = 5


class TestWindow:
    # provide a data structure to contain stats collected on test data

    def __init__(self, constraints_fns, logger):
        self.constrains_names = constraints_fns + ["_all"]
        self.logger = logger
        self.D_losses = []
        self.avg_constrains_stats = []
        # TODO fixing the following typo will invalidate all the experiments :(
        self.perfect_constrains_stats = []

    def add_constraints_stats(self, epoch, averages, percentages):
        averages = np.round(averages, 4)
        percentages = np.round(percentages, 4)
        self.logger.debug("Updating test window with averages/percentages on "
                          "constraints compliance")
        self.avg_constrains_stats.append((epoch, averages))
        self.perfect_constrains_stats.append((epoch, percentages))

    def add_losses(self, epoch, D_loss):
        self.D_losses.append((epoch, D_loss))

    def print_stats(self):
        self.logger.info("Test window collected stats")
        self.logger.info("D_losses")
        self.logger.info(self.D_losses)
        self.logger.info("Averages on constraints: {}".format(
            self.constrains_names))
        self.logger.info(self.avg_constrains_stats)
        self.logger.info("Perfect % of constraints: {}".format(
            self.constrains_names))
        self.logger.info(self.perfect_constrains_stats)


class ValidationWindow:
    # provide a data structure to contain stats collected on validation data

    # this window keeps track of the best results achieved by the ANN and can
    # thus be used to design an early-stopping criterion or other heuristics
    # during training

    def __init__(self, window_size, constraints_fns, logger):
        self.D_losses = []
        self.G_losses = []
        self.window_size = window_size
        self.items = 0
        self.constrains_names = constraints_fns + ["_all"]
        self.logger = logger
        num_constraints = len(constraints_fns)
        # initialize averages on constraints compliance with maximum error 1
        self.best_averages = num_constraints * [float(1)]
        # initialize percentages of satisfied items with 0
        # (+1 is for the percentage of items satisfying ALL constraints)
        self.best_percentages = (num_constraints + 1) * [float(0)]
        self.validation_epochs_with_no_improvement = 0
        self.best_epoch = None

    def add_constraints_stats(self, curr_epoch, averages, percentages):
        averages = np.round(averages, 4)
        percentages = np.round(percentages, 4)
        self.logger.debug("Updating validation window with "
                          "averages/percentages on constraints compliance")
        msg = "Current best {} values: {}"
        self.logger.debug(msg.format("averages", self.best_averages))
        self.logger.debug(msg.format("percentages", self.best_percentages))
        self.logger.debug("Adding averages {}...".format(averages))
        self.logger.debug("Adding percentages {}...".format(percentages))
        no_improvement = True

        for j in range(len(averages)):
            if averages[j] < self.best_averages[j]:
                self.logger.debug("Improved E[{}] from {} to {}".format(
                    self.constrains_names[j], self.best_averages[j],
                    averages[j]))
                self.best_averages[j] = averages[j]
                no_improvement = False
                self.best_epoch = curr_epoch

        for j in range(len(percentages)):
            if percentages[j] > self.best_percentages[j]:
                self.logger.debug(
                    "Improved perfect({}) from {}% to {}%".format(
                        self.constrains_names[j], self.best_percentages[j],
                        percentages[j]))
                self.best_percentages[j] = percentages[j]
                no_improvement = False
                self.best_epoch = curr_epoch

        self.logger.debug(msg.format("averages", self.best_averages))
        self.logger.debug(msg.format("percentages", self.best_percentages))

        if no_improvement:
            self.validation_epochs_with_no_improvement += 1
        else:
            self.validation_epochs_with_no_improvement = 0

    def add_losses(self, D_loss, G_loss):
        if self.items < self.window_size:
            self.D_losses.append(D_loss)
            self.G_losses.append(G_loss)
        else:
            self.D_losses[self.items % self.window_size] = D_loss
            self.G_losses[self.items % self.window_size] = G_loss
        self.items += 1

    def reset(self):
        self.D_losses = []
        self.G_losses = []
        self.items = 0
        self.validation_epochs_with_no_improvement = 0

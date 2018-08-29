import glob
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

from collections import defaultdict
from shutil import unpack_archive

DPI = 600


def load_test_window(experiment_archive):
    experiment_folder = experiment_archive.replace(".zip", os.sep)
    if not os.path.isdir(experiment_folder):
        print("Missing folder {}\nExtracting {}...".format(
            experiment_folder, experiment_archive))
        unpack_archive(experiment_archive, extract_dir=experiment_folder)
    # list available checkpoints and pick the one from the latest epoch
    _, dirs, _ = next(os.walk(experiment_folder))
    dirs.sort()
    last_epoch = experiment_folder + dirs[-1] + "/"
    return utils.undill_binary_file(last_epoch + "test_stats.dill")


def get_colours():
    colours = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
    temp = colours[2]
    colours[2] = colours[5]
    colours[5] = temp
    return ["#546380", "#b24955", "#71747b", "#78914a"] + colours


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def bar_plot(data, y_label, y_lim, title, output_path):
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = "Roboto"
    bar_width = 0.60
    plot_size = tuple(plt.rcParams["figure.figsize"])
    if len(data) > 3:
        space = (len(data) - 3) * (bar_width + 0.3)
        plot_size = (plot_size[0] + space, plot_size[1])
    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(111)
    fs = 15  # font size
    ticks_fs = 15
    label_fs = 10
    colours = get_colours()

    label_format = "{hours}h {minutes}min"

    bars = []
    for j, bar_data in enumerate(data):
        _, quartiles = bar_data
        value = quartiles[1]
        min_err = value - quartiles[0]
        max_err = quartiles[2] - value
        # do not show error if too small
        if min_err < y_lim / 15 and max_err < y_lim / 15:
            bar = ax.bar(bar_width * j, value, bar_width, color=colours[j])
        else:
            deltas = np.asarray([[min_err], [max_err]])
            bar = ax.bar(
                bar_width * j, value, bar_width, color=colours[j], yerr=deltas,
                error_kw=dict(elinewidth=3))
        bar_label = strfdelta(timedelta(minutes=value), label_format)
        ax.text(bar_width * j - bar_width / 2, bar[0].get_height() + 1,
                bar_label, ha='left', va='bottom')
        bars.append(bar)

    plt.title(title)
    ax.set_ylim(0, y_lim)
    ax.set_ylabel(y_label, fontsize=fs)
    plt.yticks(fontsize=ticks_fs, rotation=0)
    ax.set_xticks([bar_width * j for j in range(len(data))])
    xtickNames = ax.set_xticklabels([p[0] for p in data])
    plt.setp(xtickNames, fontsize=label_fs)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.show()
    plt.gcf().clear()


def function_plot(data, x_label, x_lim, y_label, y_lim, title, output_path):
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = "Roboto"
    fs = 15  # font size
    ticks_fs = 15
    alpha = 0.35
    lw = 2.5
    colours = get_colours()

    for j, curve_data in enumerate(data):
        label, xs, lows, mids, highs = curve_data
        plt.plot(xs, mids, "-", label=label, linewidth=lw, color=colours[j])
        plt.fill_between(xs, lows, highs, alpha=alpha, linewidth=0, color=colours[j])

    plt.title(title)
    plt.legend(prop={'size': fs})  # loc="upper left"
    plt.xlabel(x_label, fontsize=fs)
    plt.ylabel(y_label, fontsize=fs)
    plt.xticks(fontsize=ticks_fs, rotation=0)
    plt.yticks(fontsize=ticks_fs, rotation=0)
    plt.xlim([0, x_lim])
    plt.ylim([0, y_lim])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.show()
    plt.gcf().clear()


def get_quartiles_splits(values_by_epoch):
    percentiles = [25, 50, 75]
    xs, lows, mids, highs = [], [], [], []
    for ep, values in values_by_epoch.items():
        xs.append(ep)
        quartiles = list(np.percentile(values, percentiles))
        lows.append(quartiles[0])
        mids.append(quartiles[1])
        highs.append(quartiles[2])
    return xs, lows, mids, highs


def plot_perfect_objects_percentage(results, labels, title, output_file):
    plot_data = []
    for j, group_data in enumerate(results):
        _, group_results = group_data
        runs_results = [result for experiment, result in group_results]
        perfect_objects_by_epoch_over_runs = defaultdict(list)
        for results in runs_results:
            for ep, values in results.perfect_constrains_stats:
                perfect_objects_by_epoch_over_runs[ep].append(values[-1])
        xs, lows, mids, highs = get_quartiles_splits(
            perfect_objects_by_epoch_over_runs)
        runs_label = labels[j] + " [{} run(s)]".format(len(runs_results))
        plot_data.append((runs_label, xs, lows, mids, highs))
    max_ep = max([max(p[1]) for p in plot_data])
    function_plot(plot_data, "epoch", max_ep, "perfect objects (%)", 100,
                  title, output_file + "_perfect.png")
    if len(plot_data) == 2:
        delta_label = "_".join(labels) + "_delta"
        items = len(plot_data[1][1])
        null_values = [0] * items
        delta_xs = plot_data[1][1]
        # delta_lows = [max(0, plot_data[1][2][j] + plot_data[0][2][j]) for j in range(items)]
        delta_mids = [max(0, plot_data[1][3][j] - plot_data[0][3][j]) for j in range(items)]
        # delta_highs = [max(0, plot_data[1][4][j] + plot_data[0][4][j]) for j in range(items)]
        delta_data = [(delta_label, delta_xs, null_values, delta_mids, null_values)]
        function_plot(delta_data, "epoch", max_ep, "perfect objects delta (%)",
                      None, title, output_file + "_perfect_delta.png")


def plot_constraints_averages(results, labels, title, output_file):
    plot_data = []
    for j, group_data in enumerate(results):
        _, group_results = group_data
        runs_results = [result for experiment, result in group_results]
        perfect_constraints_by_epoch_over_runs = defaultdict(list)
        for results in runs_results:
            for ep, values in results.perfect_constrains_stats:
                perfect_constraints_by_epoch_over_runs[ep].append(
                    np.average(values[0:-1]))
        xs, lows, mids, highs = get_quartiles_splits(
            perfect_constraints_by_epoch_over_runs)
        runs_label = labels[j] + " [{} run(s)]".format(len(runs_results))
        plot_data.append((runs_label, xs, lows, mids, highs))
    max_ep = max([max(p[1]) for p in plot_data])
    function_plot(
        plot_data, "epoch", max_ep, "avg constraints satisfaction (%)",
        100, title, output_file + "_avg_constraints_satisfaction.png")
    if len(plot_data) == 2:
        delta_label = "_".join(labels) + "_delta"
        items = len(plot_data[1][1])
        null_values = [0] * items
        delta_xs = plot_data[1][1]
        # delta_lows = [max(0, plot_data[1][2][j] + plot_data[0][2][j]) for j in range(items)]
        delta_mids = [max(0, plot_data[1][3][j] - plot_data[0][3][j]) for j in range(items)]
        # delta_highs = [max(0, plot_data[1][4][j] + plot_data[0][4][j]) for j in range(items)]
        delta_data = [(delta_label, delta_xs, null_values, delta_mids, null_values)]
        function_plot(
            delta_data, "epoch", max_ep,
            "avg constraints satisfaction delta (%)",
            None, title,
            output_file + "_avg_constraints_satisfaction_delta.png")


def time_from_line(line):
    return line.split("]", maxsplit=1)[0].lstrip("[")


def log_time(log_file):
    date_pattern = "%d/%m/%Y %H:%M:%S"  # keep consistent with utils.py
    with open(log_file, "r") as log_f:
        lines = log_f.readlines()
        initial_time = time_from_line(lines[0])
        final_time = time_from_line(lines[-1])
        initial_date = datetime.strptime(initial_time, date_pattern)
        final_date = datetime.strptime(final_time, date_pattern)
        delta = final_date - initial_date
        if delta.days == 1:
            delta -= timedelta(days=1)
        return delta.total_seconds() / 60


def plot_execution_times(times, labels, title, output_file):
    percentiles = [25, 50, 75]
    plot_data = []
    max_value = 0
    for j, group_data in enumerate(times):
        _, group_results = group_data
        values = [exec_time for experiment, exec_time in group_results]
        quartiles = list(np.percentile(values, percentiles))
        runs_label = labels[j] + " [{} run(s)]".format(len(values))
        plot_data.append((runs_label, quartiles))
        max_value = max(max_value, max(values))
    bar_plot(plot_data, "execution time (min)",
             max_value + max_value / 10, title, output_file + "_time.png")


def main(args):
    plots_folder = "out/plots/"
    os.makedirs(plots_folder, exist_ok=True)
    args.output = plots_folder + args.output

    if not args.show:
        print("Disabling matplotlib interactive mode...")
        plt.ion()

    data_results = []
    execution_times = []
    for experiment in args.experiments:
        archives = sorted(glob.iglob("out/model_checkpoints/{}*.zip".format(experiment)))
        logs = sorted(glob.iglob("out/log/{}*.log".format(experiment)))
        data_results.append((experiment, [(a, load_test_window(a))
                                          for a in archives]))
        execution_times.append((experiment, [(l, log_time(l)) for l in logs]))
    plot_perfect_objects_percentage(
        data_results, args.labels, args.title, args.output)
    plot_constraints_averages(
        data_results, args.labels, args.title, args.output)
    plot_execution_times(
        execution_times, args.labels, args.title, args.output)
    for f in glob.iglob("*.log"):
        os.remove(f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiments", type=str, nargs='+',
                        required=True, help="Experiments checkpoints prefix")
    parser.add_argument("-l", "--labels", type=str, nargs='+',
                        required=True, help="Experiments plot labels")
    parser.add_argument("-t", "--title", type=str, required=True,
                        help="Plot title")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output file name")
    parser.add_argument("-s", "--show", action='store_true', help="Show plots")

    # python plot.py -e bgan_S can_S -l BGANs CANs -t "D: poly20, C: area_convex" -o D_poly20_C_area_convex_N_bgan_can
    # python plot.py -e bgan_pc_S can_pc_S can_pc_ll_S -l BGANs CANs CANS_out -t "D: poly20_pc, C: some_pc" -o D_poly20_pc_C_some_pc_N_bgan_can_canout
    # python plot.py -e bgan_pc_S can_pc_S bgan_pc_long_S can_pc_long_S -l BGANs CANs BGANs_long CANs_long -t "D: poly20_pc, C: some_pc" -o D_poly20_pc_C_some_pc_N_bgan_can_bganlong_canlong

    main(parser.parse_args())

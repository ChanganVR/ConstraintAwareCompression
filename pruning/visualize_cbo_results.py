from __future__ import print_function
from __future__ import division
import sys
import os
import glob
import argparse
import matplotlib as mpl
from collections import defaultdict
from utils import calculate_alexnet_compression_rate, read_log, Log


def find_min_objective(logs, constraint, constrained_bo):
    min_log = None
    min_iter = 1
    if constrained_bo:
        for i, log in enumerate(logs):
            if log.latency < constraint and (min_log is None or log.objective_value < min_log.objective_value):
                min_log = log
                min_iter = i
        print('Find min objective under constraint {:.2f} in iteration {}'.format(constraint, min_iter))
        print(min_log)
    else:
        for i, log in enumerate(logs):
            if min_log is None or log.objective_value < min_log.objective_value:
                min_log = log
                min_iter = i
        print('Find min objective in iteration {}'.format(min_iter))
        print(min_log)


def plot_accuracy_latency(logs, constraint, constrained_bo, title=None, saturation=False, accuracy_range=None, prefix=None):
    fig, ax = plt.subplots()
    latencies = [log.latency for log in logs]
    accuracies = [log.accuracy for log in logs]
    if constrained_bo:
        hline = ax.hlines(constraint, xmin=0, xmax=max(accuracies), linestyles='dashed', colors='blue')
        hline.set_label('Current constraint: {:.0f}'.format(int(constraint)))
    ax.legend()
    if not saturation:
        ax.plot(accuracies, latencies, 'ro')
    else:
        ax.scatter(accuracies, latencies, c=list(range(len(latencies))), cmap='Reds')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Latency(ms)')
    if not title:
        ax.set_title('Latency vs Accuracy')
    else:
        ax.set_title(title)
    if accuracy_range is not None:
        ax.set_xlim(accuracy_range)
    # plt.ylim([500, 2300])
    if prefix is not None:
        plt.savefig(prefix + '_accuracy_latency.png')
    plt.show()


def plot_accuracy_latency_ratio(logs, title=None, saturation=False):
    ratios = [log.latency_ratio for log in logs]
    accuracies = [log.accuracy for log in logs]
    saturation_degrees = [log.sampling_time for log in logs]
    if not saturation:
        plt.plot(accuracies, ratios, 'ro')
    else:
        plt.scatter(accuracies, ratios, c=saturation_degrees, cmap='Reds')
    plt.xlabel('Accuracy')
    plt.ylabel('Latency ratio(%)')
    if not title:
        plt.title('Latency ratio vs Accuracy')
    else:
        plt.title(title)
    plt.xlim([0, 0.6])
    # plt.ylim([500, 2300])
    plt.show()


def area_under_curve_diff(logs, original_latency, accuracy_range=(0, 0.5), bin_width=0.01):
    # calculate the AUC difference for the last sampled set of parameters
    # the problem with using AUC is that with the same sampled parameter, the objective may be different,
    # which depends on the history of sampled parameters
    # the problem of using AUC difference is, as BO goes on, AUC difference is smaller and smaller
    # and is not the same as our real objective
    # Both of them violate the bayesian process assumption (continuity)
    last_area = area_under_curve(logs[:-1], original_latency, accuracy_range, bin_width)
    current_area = area_under_curve(logs, original_latency, accuracy_range, bin_width)
    return last_area - current_area


def area_under_curve(logs, upper_bound, accuracy_range=(0, 0.55), bin_width=0.01, ratio=True):
    # accuracy_dict stores accuracy range as key and its corresponding latencies as value
    accuracy_dict = defaultdict(list)
    for log in logs:
        if accuracy_range[0] <= log.accuracy <= accuracy_range[1]:
            key = int((log.accuracy-accuracy_range[0])/bin_width)
            accuracy_dict[key].append(log)

    area = 0
    for i in range(int((accuracy_range[1]-accuracy_range[0])/bin_width)):
        # even the bin doesn't have any points
        if i not in accuracy_dict:
            area += bin_width * upper_bound
        else:
            if ratio:
                area += bin_width * min(accuracy_dict[i], key=Log.get_ratio).latency_ratio
            else:
                area += bin_width * min(accuracy_dict[i], key=Log.get_latency).latency
    return area


def range_distribution(logs, accuracy_range=(0, 0.55), bin_width=0.1):
    # number of points falling in each accuracy bin
    accuracy_dict = defaultdict(list)
    for log in logs:
        bin_num = int((log.accuracy-accuracy_range[0])/bin_width)
        accuracy_dict[bin_num].append(log)

    for i in range(int((accuracy_range[1] - accuracy_range[0])/bin_width)):
        print('Range [{}, {}]: {}'.format(bin_width*i, bin_width*(i+1), len(accuracy_dict[i])))


def find_best_logs(logs, accuracy_range=(0, 0.55), bin_width=0.01):
    # search for each accuracy bin with minimal latency
    accuracy_dict = defaultdict(list)
    for log in logs:
        if accuracy_range[0] <= log.accuracy <= accuracy_range[1]:
            key = int((log.accuracy-accuracy_range[0])/bin_width)
            accuracy_dict[key].append(log)
    best_logs = [min(accuracy_dict[key], key=Log.get_latency) for key in accuracy_dict]

    return best_logs


def plot_lower_bound_curve(logs):
    bin_width = 0.01
    best_logs = find_best_logs(logs, bin_width=bin_width)
    plot_accuracy_latency(best_logs, title='Lower bound curve with search bin width {}'.format(bin_width))


def plot_latency_compression_curve(logs):
    # first find the best pruning parameter for a given accuracy bin
    bin_width = 0.01
    best_logs = find_best_logs(logs, bin_width=bin_width)

    # plots the graph
    latencies = [log.latency for log in best_logs]
    compression_rates = [calculate_alexnet_compression_rate(log.pruning_dict) for log in best_logs]
    plt.plot(compression_rates, latencies, 'ro')
    plt.xlabel('Compression rate')
    plt.ylabel('Latency(ms)')
    plt.title('Latency vs Compression rate with bin width {}'.format(bin_width))
    plt.xlim([0, 1])
    plt.ylim([500, 2300])
    plt.show()


def plot_uac_iteration(logs, upper_bound, accuracy_range=(0, 0.55), bin_width=0.01, diff=False, prefix=None):
    iterations = []
    uacs = []
    for i, log in enumerate(logs):
        iterations.append(i)
        uacs.append(area_under_curve(logs[:i + 1], upper_bound, accuracy_range, bin_width))
    if diff:
        temp = []
        for i in range(len(uacs)-1):
            temp.append(uacs[i] - uacs[i+1])
        iterations = iterations[1:]
        uacs = temp

    plt.plot(iterations, uacs, 'b')
    plt.xlabel('Iterations')
    plt.ylabel('AUC')
    plt.title('AUC vs iterations with bin width {}'.format(bin_width))

    if prefix is not None:
        plt.savefig(prefix + '_uac_iteration.png')
    plt.show()


def plot_objective_time(logs, constraint, constrained_bo, prefix=None):
    fig, ax = plt.subplots()
    objective_values = [0]
    for log in logs:
        if constrained_bo:
            if log.objective_value < objective_values[-1] and log.latency < constraint:
                objective_values.append(log.objective_value)
            else:
                objective_values.append(objective_values[-1])
        else:
            if log.objective_value < objective_values[-1]:
                objective_values.append(log.objective_value)
            else:
                objective_values.append(objective_values[-1])
    iterations = list(range(len(objective_values)))
    ax.plot(iterations, objective_values)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective values')
    ax.set_title('Objective values vs iterations')

    if prefix is not None:
        plt.savefig(prefix + '_objective_time.png')
    plt.show()


def main(file_path):
    if os.path.isdir(file_path):
        files = sorted(glob.glob(os.path.join(file_path, '*bo.log')))
        log_dir = file_path
    else:
        files = [file_path]
        log_dir = os.path.dirname(file_path)
    fig_dir = os.path.join(log_dir, 'plots')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    for log_file in files:
        print('\nPlot', log_file)
        logs, constraint = read_log(log_file)
        prefix = os.path.join(fig_dir, os.path.basename(log_file)[:-4])
        if constraint is None:
            constrained_bo = False
        else:
            constrained_bo = True
        print('Number of iterations:', len(logs))
        if constraint is not None:
            print('Current constraint:', constraint)
        find_min_objective(logs, constraint, constrained_bo)
        # print('Area under curve with range ({}, {}) is {}'.format(0, 0.55, area_under_curve(logs, 1, (0, 0.55))))
        # range_distribution(logs)
        plot_accuracy_latency(logs, constraint, constrained_bo, saturation=True, prefix=prefix)
        # plot_latency_compression_curve(res)
        # plot_lower_bound_curve(res)
        # plot_uac_iteration(logs, 1, prefix=prefix)
        plot_objective_time(logs, constraint, constrained_bo, prefix=prefix)
        # plot_accuracy_latency_ratio(res, saturation=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize cfp output logs')
    parser.add_argument('file_path')
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()
    if not args.display:
        mpl.use('Agg')
    from matplotlib import pyplot as plt
    main(args.file_path)

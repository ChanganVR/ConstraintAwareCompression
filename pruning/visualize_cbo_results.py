from __future__ import print_function
from __future__ import division
import sys
import os
import glob
import argparse
import matplotlib as mpl
from collections import defaultdict


def find_min_objective(logs, constraint, constrained_bo):
    min_log = None
    min_obj = 0
    if constrained_bo:
        for i, log in enumerate(logs):
            if log.latency < constraint and log.objective_value < min_obj:
                min_obj = log.objective_value
                min_log = log
    else:
        for i, log in enumerate(logs):
            if log.objective_value < min_obj:
                min_obj = log.objective_value
                min_log = log
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
        # ax.set_title('Latency vs Accuracy')
        pass
    else:
        ax.set_title(title)
    if accuracy_range is not None:
        ax.set_xlim(accuracy_range)
    # plt.ylim([500, 2300])
    if prefix is not None:
        plt.savefig(prefix + '_accuracy_latency.png')
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


def plot_layerwise_pruning_param(logs, prefix=None):
    layers = logs[0].pruning_dict.keys()
    for layer in layers:
        vals = [log.pruning_dict[layer] for log in logs]
        iters = range(len(logs))
        fig, ax = plt.subplots()
        ax.plot(iters, vals, 'o')
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Pruning percentage')
        # ax.set_title('Percentage vs iteration in {}'.format(layer))
        if prefix is not None:
            plt.savefig(prefix + '_{}.png'.format(layer))
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
    layerwise_pruning_layers = raw_input('Plot layerwise pruning parameter layers: ')
    for i, log_file in enumerate(files):
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
        plot_accuracy_latency(logs, constraint, constrained_bo, saturation=True, prefix=prefix)
        plot_objective_time(logs, constraint, constrained_bo, prefix=prefix)
        if layerwise_pruning_layers == 'all' or str(i) in layerwise_pruning_layers:
            plot_layerwise_pruning_param(logs, prefix=prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize cfp output logs')
    parser.add_argument('file_path')
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()
    if not args.display:
        mpl.use('Agg')
    from matplotlib import pyplot as plt
    from utils import read_log
    main(args.file_path)

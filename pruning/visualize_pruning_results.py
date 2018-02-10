from __future__ import print_function
from __future__ import division
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import calculate_alexnet_compression_rate, read_log, Log, read_fp_log


def find_max_objective(logs):
    max_log = logs[0]
    max_iter = 1
    for iter, log in enumerate(logs):
        if log.latency_ratio < max_log.latency_ratio:
            max_log = log
            max_iter = iter
    print('Find max objective value in iteration', max_iter)
    print(max_log)


def plot_accuracy_latency(logs, title=None, saturation=False, accuracy_range=None):
    latencies = [log.latency for log in logs]
    accuracies = [log.accuracy for log in logs]
    if not saturation:
        plt.plot(accuracies, latencies, 'ro')
    else:
        plt.scatter(accuracies, latencies, c=list(range(len(latencies))), cmap='Reds')
    plt.xlabel('Accuracy')
    plt.ylabel('Latency(ms)')
    if not title:
        plt.title('Latency vs Accuracy')
    else:
        plt.title(title)
    if accuracy_range is not None:
        plt.xlim(accuracy_range)
    # plt.ylim([500, 2300])
    plt.show()


def plot_accuracy_latency_ratio(logs, title=None, saturation=False):
    ratios = [log.latency_ratio for log in logs]
    accuracies = [log.accuracy for log in logs]
    saturations = [log.sampling_time for log in logs]
    if not saturation:
        plt.plot(accuracies, ratios, 'ro')
    else:
        plt.scatter(accuracies, ratios, c=saturations, cmap='Reds')
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


def plot_uac_vs_iteration(logs, upper_bound, accuracy_range=(0, 0.55), bin_width=0.01, diff=False):
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
    plt.show()


def plot_objective_time(logs, constraint=None, min_obj=True):
    if min_obj:
        objective_values = [0]
        for log in logs:
            if constraint is not None:
                if log.objective_value < objective_values[-1] and log.latency < constraint:
                    objective_values.append(log.objective_value)
                else:
                    objective_values.append(objective_values[-1])
    else:
        objective_values = [logs[0].objective_value]
        for log in logs[1:]:
            if log.objective_value > objective_values[-1]:
                objective_values.append(log.objective_value)
            else:
                objective_values.append(objective_values[-1])
    iterations = list(range(len(objective_values)))
    plt.plot(iterations, objective_values)
    plt.xlabel('Iterations')
    plt.ylabel('Objective values')
    plt.title('Objective values vs iterations')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Log file needs to be specified')
    elif len(sys.argv) == 2:
        log_results, constraint = read_log(sys.argv[1])
    elif len(sys.argv) == 3:
        log_results = read_fp_log(sys.argv[1], int(sys.argv[2]))
    else:
        raise ValueError('Input arguments format is wrong')
    print('Number of iterations:', len(log_results))
    print('Constraint:', constraint)
    find_max_objective(log_results)
    print('Area under curve with range ({}, {}) is {}'.format(0, 0.55, area_under_curve(log_results, 1, (0, 0.55))))
    range_distribution(log_results)
    plot_accuracy_latency(log_results, saturation=True)
    # plot_latency_compression_curve(res)
    # plot_lower_bound_curve(res)
    plot_uac_vs_iteration(log_results, 1)
    plot_objective_time(log_results, constraint)
    # plot_accuracy_latency_ratio(res, saturation=True)

# sample pruning log
# INFO:root:=================================>>>Pruning starts<<<=================================
# INFO:root:conv1		conv2		conv3		conv4		conv5		fc6			fc7			fc8
# INFO:root:0.12		0.19		0.55		0.15		0.42		0.05		0.24		0.54
# INFO:root:Pruning takes(s):              6.23
# INFO:root:Testing latency takes(s):      9.90
# INFO:root:Latency(ms):                   1760.99
# INFO:root:Testing accuracy takes(s):     9.42
# INFO:root:Accuracy:                      0.54
# INFO:root:Total time(s):                 25.56
# INFO:root:Objective value:               65.82

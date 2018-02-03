from __future__ import print_function
from __future__ import division
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import calculate_alexnet_compression_rate, read_log, Result


def find_max_objective(results):
    max_res = results[0]
    max_iter = 1
    for iter, res in enumerate(results):
        if res.latency_ratio < max_res.latency_ratio:
            max_res = res
            max_iter = iter
    print('Find max objective value in iteration', max_iter)
    print(max_res)


def plot_accuracy_latency(results, title=None, saturation=False):
    latencies = [res.latency for res in results]
    accuracies = [res.accuracy for res in results]
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
    plt.xlim([0, 0.6])
    # plt.ylim([500, 2300])
    plt.show()


def plot_accuracy_latency_ratio(results, title=None, saturation=False):
    ratios = [res.latency_ratio for res in results]
    accuracies = [res.accuracy for res in results]
    saturations = [res.sampling_time for res in results]
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


def area_under_curve_diff(results, original_latency, accuracy_range=(0, 0.5), bin_width=0.01):
    # calculate the AUC difference for the last sampled set of parameters
    # the problem with using AUC is that with the same sampled parameter, the objective may be different,
    # which depends on the history of sampled parameters
    # the problem of using AUC difference is, as BO goes on, AUC difference is smaller and smaller
    # and is not the same as our real objective
    # Both of them violate the bayesian process assumption (continuity)
    last_area = area_under_curve(results[:-1], original_latency, accuracy_range, bin_width)
    current_area = area_under_curve(results, original_latency, accuracy_range, bin_width)
    return last_area - current_area


def area_under_curve(results, upper_bound, accuracy_range=(0, 0.55), bin_width=0.01, ratio=True):
    # accuracy_dict stores accuracy range as key and its corresponding latencies as value
    accuracy_dict = defaultdict(list)
    for result in results:
        if accuracy_range[0] <= result.accuracy <= accuracy_range[1]:
            key = int((result.accuracy-accuracy_range[0])/bin_width)
            accuracy_dict[key].append(result)

    area = 0
    for i in range(int((accuracy_range[1]-accuracy_range[0])/bin_width)):
        # even the bin doesn't have any points
        if i not in accuracy_dict:
            area += bin_width * upper_bound
        else:
            if ratio:
                area += bin_width * min(accuracy_dict[i], key=Result.get_ratio).latency_ratio
            else:
                area += bin_width * min(accuracy_dict[i], key=Result.get_latency).latency
    return area


def range_distribution(results, accuracy_range=(0, 0.55), bin_width=0.1):
    # number of points falling in each accuracy bin
    accuracy_dict = defaultdict(list)
    for result in results:
        bin_num = int((result.accuracy-accuracy_range[0])/bin_width)
        accuracy_dict[bin_num].append(result)

    for i in range(int((accuracy_range[1] - accuracy_range[0])/bin_width)):
        print('Range [{}, {}]: {}'.format(bin_width*i, bin_width*(i+1), len(accuracy_dict[i])))


def find_best_results(results, accuracy_range=(0, 0.55), bin_width=0.01):
    # search for each accuracy bin with minimal latency
    accuracy_dict = defaultdict(list)
    for result in results:
        if accuracy_range[0] <= result.accuracy <= accuracy_range[1]:
            key = int((result.accuracy-accuracy_range[0])/bin_width)
            accuracy_dict[key].append(result)
    best_results = [min(accuracy_dict[key], key=Result.get_latency) for key in accuracy_dict]

    return best_results


def plot_lower_bound_curve(results):
    bin_width = 0.01
    best_results = find_best_results(results, bin_width=bin_width)
    plot_accuracy_latency(best_results, title='Lower bound curve with search bin width {}'.format(bin_width))


def plot_latency_compression_curve(results):
    # first find the best pruning parameter for a given accuracy bin
    bin_width = 0.01
    best_results = find_best_results(results, bin_width=bin_width)

    # plots the graph
    latencies = [res.latency for res in best_results]
    compression_rates = [calculate_alexnet_compression_rate(res.pruning_dict) for res in best_results]
    plt.plot(compression_rates, latencies, 'ro')
    plt.xlabel('Compression rate')
    plt.ylabel('Latency(ms)')
    plt.title('Latency vs Compression rate with bin width {}'.format(bin_width))
    plt.xlim([0, 1])
    plt.ylim([500, 2300])
    plt.show()


def plot_uac_vs_iteration(results, upper_bound, accuracy_range=(0, 0.55), bin_width=0.01, diff=False):
    iterations = []
    uacs = []
    for i, result in enumerate(results):
        iterations.append(i)
        uacs.append(area_under_curve(results[:i+1], upper_bound, accuracy_range, bin_width))
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Log file needs to be specified')
        exit(1)
    else:
        res = []
        for log_file in sys.argv[1:]:
            res += read_log(sys.argv[1])
    print('Number of iterations:', len(res))
    find_max_objective(res)
    print('Area under curve with range ({}, {}) is {}'.format(0, 0.55, area_under_curve(res, 1, (0, 0.55))))
    range_distribution(res)
    # plot_accuracy_latency(res, saturation=True)
    # plot_latency_compression_curve(res)
    # plot_lower_bound_curve(res)
    # plot_uac_vs_iteration(res, 1)
    plot_accuracy_latency_ratio(res, saturation=True)

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

from __future__ import print_function
from __future__ import division
import sys
import matplotlib.pyplot as plt
from collections import defaultdict


class Result(object):
    def __init__(self, pruning_dict, pruning_time, testing_latency_time, latency, testing_accuracy_time,
                 accuracy, total_time, objective_value):
        self.pruning_dict = pruning_dict
        self.pruning_time = pruning_time
        self.testing_latency_time = testing_latency_time
        self.latency = latency
        self.testing_accuracy_time = testing_accuracy_time
        self.accuracy = accuracy
        self.total_time = total_time
        self.objective_value = objective_value

    def __str__(self):
        string = 'conv1\tconv2\tconv3\tconv4\tconv5\tfc6\tfc7\tfc8' + '\n'
        pruning_percentages = '{conv1} {conv2} {conv3} {conv4} {conv5} {fc6} {fc7} {fc8}'.format(**self.pruning_dict)
        string += '\t'.join(['%.2f' % float(x) for x in pruning_percentages.split()]) + '\n'
        string += "{:<20} {:.2f}".format('Latency:', self.latency) + '\n'
        string += "{:<20} {:.2f}".format('Accuracy:', self.accuracy) + '\n'
        string += "{:<20} {:.2f}".format('Objective:', self.objective_value)
        return string

    @staticmethod
    def get_latency(result):
        return result.latency


def read_log(log_file):
    results = []
    with open(log_file) as fo:
        lines = fo.readlines()
    if len(lines) == 0:
        raise IOError('Can not read log file')
    for i, line in enumerate(lines):
        # need to have a full pruning result
        if i + 9 >= len(lines):
            break
        if 'Pruning starts' in line:
            layers = [x for x in lines[i+1][10:].strip().split()]
            pruning_percentages = [float(x) for x in lines[i+2][10:].strip().split()]
            pruning_dict = {x: y for x, y in zip(layers, pruning_percentages)}
            pruning_time = float(lines[i+3].strip().split()[-1])
            testing_latency_time = float(lines[i+4].strip().split()[-1])
            latency = float(lines[i+5].strip().split()[-1])
            testing_accuracy_time = float(lines[i+6].strip().split()[-1])
            accuracy = float(lines[i+7].strip().split()[-1])
            total_time = float(lines[i+8].strip().split()[-1])
            objective_value = float(lines[i+9].strip().split()[-1])
            result = Result(pruning_dict, pruning_time, testing_latency_time, latency,
                            testing_accuracy_time, accuracy, total_time, objective_value)
            results.append(result)

    return results


def find_max_objective(results):
    max_res = results[0]
    max_iter = 1
    for iter, res in enumerate(results):
        if res.objective_value > max_res.objective_value:
            max_res = res
            max_iter = iter
    print('Find max objective value in iteration', max_iter)
    print(max_res)


def plot_accuracy_latency_curve(results):
    latencies = [res.latency for res in results]
    accuracies = [res.accuracy for res in results]
    plt.plot(accuracies, latencies, 'ro')
    plt.xlabel('Accuracy')
    plt.ylabel('Latency(ms)')
    plt.title('Accuracy vs Latency')
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


def area_under_curve(results, original_latency, accuracy_range=(0.2, 0.5), bin_width=0.01):
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
            area += bin_width * original_latency
        else:
            area += bin_width * min(accuracy_dict[i], key=Result.get_latency).latency
    return area


if __name__ == '__main__':
    res = read_log(sys.argv[1])
    print('Number of iterations:', len(res))
    find_max_objective(res)
    print(area_under_curve(res, 2352))
    plot_accuracy_latency_curve(res)


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

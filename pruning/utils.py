from __future__ import division
from __future__ import print_function

import logging
import os
import re
import sys

import numpy as np
from matplotlib import pyplot as plt


class Log(object):
    def __init__(self, pruning_dict, pruning_time, testing_latency_time, latency, testing_accuracy_time,
                 compression_rate, accuracy, total_time, objective_value, speedup, sampled_iter):
        self.pruning_dict = pruning_dict
        self.pruning_time = pruning_time
        self.testing_latency_time = testing_latency_time
        self.latency = latency
        self.testing_accuracy_time = testing_accuracy_time
        self.compression_rate = compression_rate
        self.accuracy = accuracy
        self.total_time = total_time
        self.objective_value = objective_value
        # sparse / original conv
        self.latency_ratio = speedup
        self.sampled_iteration = sampled_iter

    def __str__(self):
        layers = [layer for layer, _ in sorted(self.pruning_dict.items())]
        pruning_percentages = [percent for _, percent in sorted(self.pruning_dict.items())]
        string = '\n{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(*layers) + '\n'
        string += '{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}'.format(*pruning_percentages) + '\n'
        string += "{:<20} {:.2f}".format('Latency:', self.latency) + '\n'
        string += "{:<20} {:.4f}".format('Compression rate:', self.compression_rate) + '\n'
        string += "{:<20} {:.4f}".format('Accuracy:', self.accuracy) + '\n'
        string += "{:<20} {:.4f}".format('Objective:', self.objective_value) + '\n'
        string += "{:<20} {:.2f}".format('Latency ratio:', self.latency_ratio) + '\n'
        string += "{:<20} {:.2f}".format('Sampled iteration:', self.sampled_iteration)
        return string

    @staticmethod
    def get_latency(log):
        return log.latency

    @staticmethod
    def get_ratio(log):
        return log.latency_ratio


def find_next_phase(log_file):
    with open(log_file) as fo:
        lines = [line.strip() for line in fo.readlines()]
    if len(lines) == 0:
        raise IOError('Can not read log file')
    index = lines[-1].find('th iteration')
    if index == -1:
        raise ValueError('fine_pruning.log last line format incorrect')
    else:
        t = int(lines[-1][index-1])
    if 'Bayesian optimization' in lines[-1]:
        next_phase = 'pruning'
    elif 'Pruning the best sampled model' in lines[-1]:
        next_phase = 'finetuning'
    elif 'Fine-tuning' in lines[-1]:
        next_phase = 'bayesian optimization'
        t += 1
    else:
        raise ValueError('Log file format incorrect')

    return t, next_phase


def read_log(log_file):
    # read single bayesian optimization log file
    logs = []
    original_latency = 238
    with open(log_file) as fo:
        lines = [line.strip() for line in fo.readlines()]
    if len(lines) == 0:
        raise IOError('Can not read log file')

    sampling_counter = 0
    constraint = None
    error_counter = 0
    for i, line in enumerate(lines):
        # need to have a full pruning log
        if i + 5 >= len(lines):
            break
        if 'Original latency' in line:
            original_latency = float(line.split()[-1])
        if 'Current constraint' in line:
            constraint = float(line.split()[-1])
        if 'Constraint type' in line:
            constraint_type = line.split()[-1]
        if 'Pruning starts' in line:
            layers = [x for x in lines[i+1].split()[3:]]
            pruning_percentages = [float(x) for x in lines[i+2].split()[3:]]
            pruning_dict = {x: y for x, y in zip(layers, pruning_percentages)}
            if 'Fail to' in lines[i+3] or 'Fail to' in lines[i+4]:
                # if error occurs, skip this log
                error_counter += 1
                continue
            else:
                if constraint_type == 'latency':
                    latency = float(lines[i+3].split()[-1])
                    compression_rate = -1
                elif constraint_type == 'compression_rate':
                    compression_rate = float(lines[i + 3].split()[-1])
                    latency = -1
                else:
                    raise ValueError('Log file format incorrect')
                accuracy = float(lines[i + 4].split()[-1])
            objective_value = float(lines[i+5].split()[-1])
            log = Log(pruning_dict, -1, -1, latency, -1, compression_rate, accuracy, -1, objective_value,
                      latency / original_latency, sampling_counter)
            sampling_counter += 1
            logs.append(log)

    if error_counter != 0:
        logging.warning('Fail to read {} test_accuracy.txt/test_latency.txt'.format(error_counter))
    return logs, constraint


def calculate_compression_rate(caffemodel_file, prototxt_file):
    os.environ['GLOG_minloglevel'] = '2'
    import caffe

    net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)
    total_parameters = 0
    non_zeros = 0
    for layer in net.params:
        # find the absolute threshold with percentile lower than pruning_percentage
        weights = net.params[layer][0].data
        biases = net.params[layer][1].data
        total_parameters += np.ma.size(weights) + np.ma.size(biases)
        print('Layer ', layer, total_parameters)
        non_zeros += np.count_nonzero(weights) + np.count_nonzero(biases)

    return non_zeros / total_parameters


def calculate_alexnet_compression_rate(pruning_dict):
    layer_weights_dict = {'conv1': 34944, 'conv2': 342400, 'conv3': 1227520, 'conv4': 1891456, 'conv5': 2334080,
                          'fc6': 40086912, 'fc7': 56868224, 'fc8': 60965224}
    total_weights = sum(layer_weights_dict.values())
    pruned_weights = 0
    for layer in pruning_dict:
        pruned_weights += layer_weights_dict[layer] * pruning_dict[layer]

    return 1 - pruned_weights / total_weights


def plot_val_acc_in_bo_iters(log_file):
    with open(log_file) as log:
        text = log.read()
        res = re.findall(r"In bo_iter (\d+), best result has train acc (0\.\d+) and val acc (0\.\d+)", text)
        sampled_iter = []
        train_acc = []
        val_acc = []
        for r in res:
            sampled_iter.append(float(r[0]))
            train_acc.append(float(r[1]))
            val_acc.append(float(r[2]))
        print(sampled_iter, train_acc, val_acc)

    plt.scatter(sampled_iter, train_acc)
    plt.scatter(sampled_iter, val_acc)
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Bayesian optimization iterations')
    plt.ylabel('Accuracy')
    plt.title('Validation acc and train acc vs bo iterations')
    plt.show()


if __name__ == '__main__':
    plot_val_acc_in_bo_iters(sys.argv[1])


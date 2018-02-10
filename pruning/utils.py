from __future__ import print_function
from __future__ import division
import numpy as np
import os
import logging
import sys
import json


class Log(object):
    def __init__(self, pruning_dict, pruning_time, testing_latency_time, latency, testing_accuracy_time,
                 accuracy, total_time, objective_value, speedup, time):
        self.pruning_dict = pruning_dict
        self.pruning_time = pruning_time
        self.testing_latency_time = testing_latency_time
        self.latency = latency
        self.testing_accuracy_time = testing_accuracy_time
        self.accuracy = accuracy
        self.total_time = total_time
        self.objective_value = objective_value
        # sparse / original conv
        self.latency_ratio = speedup
        self.sampling_time = time

    def __str__(self):
        layers = [layer for layer, _ in sorted(self.pruning_dict.items())]
        pruning_percentages = [percent for _, percent in sorted(self.pruning_dict.items())]
        string = '\n{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(*layers) + '\n'
        string += '{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}'.format(*pruning_percentages) + '\n'
        string += "{:<20} {:.2f}".format('Latency:', self.latency) + '\n'
        string += "{:<20} {:.2f}".format('Accuracy:', self.accuracy) + '\n'
        string += "{:<20} {:.2f}".format('Objective:', self.objective_value) + '\n'
        string += "{:<20} {:.2f}".format('Latency ratio:', self.latency_ratio) + '\n'
        string += "{:<20} {:.2f}".format('Sampling time:', self.sampling_time)
        return string

    @staticmethod
    def get_latency(log):
        return log.latency

    @staticmethod
    def get_ratio(log):
        return log.latency_ratio


def read_fp_log(log_file, bo_num=None):
    # read log file for fine-pruning procedure
    logs = []
    with open(log_file) as fo:
        lines = [line.strip() for line in fo.readlines()]
    if len(lines) == 0:
        raise IOError('Can not read log file')

    # read original latency from the beginning of the file
    original_latency = [float(line.split()[-1]) for line in lines[:10] if 'Original latency' in line][0]

    # find out logs belonging to {bo_num}th bayesian optimization
    if bo_num is not None:
        start_pattern = 'Start {}th fine-pruning iteration'.format(bo_num)
        end_pattern = 'Start {}th fine-pruning iteration'.format(bo_num+1)
        boundaries = [i for i, line in enumerate(lines) if start_pattern in line or end_pattern in line]
        if len(boundaries) == 1:
            lines = lines[boundaries[0]:]
        elif len(boundaries) == 2:
            lines = lines[boundaries[0]: boundaries[1]]
        else:
            raise RuntimeError('Fail to find {}th fine-pruning iteration log'.format(bo_num))

    sampling_counter = 0
    for i, line in enumerate(lines):
        # need to have a full pruning los
        if i + 9 >= len(lines):
            break
        if 'Pruning starts' in line:
            layers = [x for x in lines[i+1][10:].split()]
            pruning_percentages = [float(x) for x in lines[i+2][10:].split()]
            pruning_dict = {x: y for x, y in zip(layers, pruning_percentages)}
            pruning_time = float(lines[i+3].split()[-1])
            testing_latency_time = float(lines[i+4].split()[-1])
            latency = float(lines[i+5].split()[-1])
            testing_accuracy_time = float(lines[i+6].split()[-1])
            accuracy = float(lines[i+7].split()[-1])
            total_time = float(lines[i+8].split()[-1])
            objective_value = float(lines[i+9].split()[-1])
            log = Log(pruning_dict, pruning_time, testing_latency_time, latency, testing_accuracy_time,
                         accuracy, total_time, objective_value, latency / original_latency, sampling_counter)
            sampling_counter += 1
            logs.append(log)

    return logs


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
    original_latency = 239
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
        if 'Bayesian optimization tradeoff factor' in line:
            sampling_counter = 0
        if 'Original latency' in line:
            original_latency = float(line.split()[-1])
        if 'Latency constraint' in line:
            constraint = float(line.split()[-1])
        if 'Pruning starts' in line:
            layers = [x for x in lines[i+1].split()[3:]]
            pruning_percentages = [float(x) for x in lines[i+2].split()[3:]]
            pruning_dict = {x: y for x, y in zip(layers, pruning_percentages)}
            if 'Fail to read' in lines[i+3] or 'Fail to read' in lines[i+4]:
                # if error occurs, skip this log
                error_counter += 1
                continue
            else:
                latency = float(lines[i+3].split()[-1])
                accuracy = float(lines[i+4].split()[-1])
            objective_value = float(lines[i+5].split()[-1])
            log = Log(pruning_dict, -1, -1, latency, -1, accuracy, -1, objective_value,
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
        total_parameters += np.ma.size(weights) + np.ma.size(weights)
        print('Layer ', layer, total_parameters)
        non_zeros += np.count_nonzero(weights) + np.count_nonzero(biases)

    return non_zeros / total_parameters


def calculate_alexnet_compression_rate(pruning_dict):
    layer_weights_dict = {'conv1': 69696, 'conv2': 684096, 'conv3': 2453568, 'conv4': 3780672, 'conv5': 4665408,
                          'fc6': 80162880, 'fc7': 113717312, 'fc8': 121909312}
    total_weights = sum(layer_weights_dict.values())
    pruned_weights = 0
    for layer in pruning_dict:
        pruned_weights += layer_weights_dict[layer] * pruning_dict[layer]

    return 1 - pruned_weights / total_weights


def create_different_sparsity(log_file, compression_levels):
    # find closest pruning parameters as compression levels
    logs = read_log(log_file)
    compression_dict = {level: None for level in compression_levels}
    for log in logs:
        compression_rate = calculate_alexnet_compression_rate(log.pruning_dict)
        for level, closest_value in compression_dict.items():
            if closest_value is None or abs(level - compression_rate) < abs(level - closest_value[1]):
                compression_dict[level] = (log, compression_rate)

    # create pruning model
    pruning_dict_file = 'results/pruning_dict.txt'
    original_prototxt_file = 'models/bvlc_reference_caffenet/train_val.prototxt'
    caffemodel_file = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    for level, (log, rate) in compression_dict.items():
        temp_caffemodel_file = 'results/alexnet_density_{}.caffemodel'.format(level)
        with open(pruning_dict_file, 'w') as fo:
            json.dump(log.pruning_dict, fo)
        command = ['python', 'pruning/prune.py', caffemodel_file, original_prototxt_file,
                   temp_caffemodel_file, pruning_dict_file]
        os.system(' '.join(command))


if __name__ == '__main__':
    # caffemodel = sys.argv[1]
    # prototxt = sys.argv[2]
    # print(calculate_compression_rate(caffemodel, prototxt))

    find_next_phase('results/fp_5_40_linear/fine_pruning.log')

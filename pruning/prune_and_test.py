from __future__ import division
import os
import numpy as np
import logging
import sys
import json
os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()


def prune_and_test(input_caffemodel, prototxt, dataset, pruning_dict):
    if dataset == 'imagenet':
        test_iters = 200
    else:
        raise NotImplementedError
    # read and prune caffemodel weights
    net = caffe.Net(prototxt, input_caffemodel, caffe.TEST)
    total_params = 0
    non_zeros = 0
    for layer in net.params:
        pruning_percentage = pruning_dict[layer]

        # find the absolute threshold with percentile lower than pruning_percentage
        weights = net.params[layer][0].data
        threshold = np.percentile(np.abs(weights), pruning_percentage * 100)
        weights[np.abs(weights) < threshold] = 0
        # does not prune the biases
        biases = net.params[layer][1].data
        # threshold = np.percentile(np.abs(biases), pruning_percentage * 100)
        # biases[np.abs(biases) < threshold] = 0

        total_params += np.ma.size(weights) + np.ma.size(biases)
        non_zeros += np.count_nonzero(weights) + np.count_nonzero(biases)

    compression_rate = non_zeros / total_params
    accuracy = 0
    for i in range(test_iters):
        net.forward()
        accuracy += net.blobs['accuracy'].data
    accuracy /= test_iters

    with open('results/prune_and_test.txt', 'w') as fo:
        fo.write(' '.join([str(compression_rate), str(accuracy)]))


if __name__ == '__main__':
    with open(sys.argv[4]) as fo:
        pruning_percentage_dict = json.load(fo)
    prune_and_test(sys.argv[1], sys.argv[2], sys.argv[3], pruning_percentage_dict)

from __future__ import print_function
import numpy as np
import sys
import json
import os
import logging
from utils import read_log


def prune(input_caffemodel, prototxt_file, output_caffemodel, pruning_dict):
    # logging.basicConfig(filename='results/prune_debug.log', filemode='w', level=logging.DEBUG)
    # surpress log output from caffe loading
    os.environ['GLOG_minloglevel'] = '2'
    import caffe
    net = caffe.Net(prototxt_file, input_caffemodel, caffe.TEST)
    for layer in net.params:
        pruning_percentage = pruning_dict[layer]

        # find the absolute threshold with percentile lower than pruning_percentage
        weights = net.params[layer][0].data
        threshold = np.percentile(np.abs(weights), pruning_percentage * 100)
        weights[np.abs(weights) < threshold] = 0

        # biases = net.params[layer][1].data
        # threshold = np.percentile(np.abs(biases), pruning_percentage * 100)
        # biases[np.abs(biases) < threshold] = 0

    # finish pruning and writing weights to temporary caffemodel
    net.save(output_caffemodel)
    # logging.info('Pruning done')


if __name__ == '__main__':
    with open(sys.argv[4]) as fo:
        pruning_percentage_dict = json.load(fo)
    prune(sys.argv[1], sys.argv[2], sys.argv[3], pruning_percentage_dict)

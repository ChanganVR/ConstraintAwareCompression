from __future__ import print_function
import numpy as np
import sys
import json
import os
import logging
from utils import read_log
from visualize_cfp_results import find_best_logs


def prune(input_caffemodel, prototxt_file, output_caffemodel, pruning_percentage_dict):
    # surpress log output from caffe loading
    os.environ['GLOG_minloglevel'] = '2'
    import caffe
    # logging.basicConfig(filename='prune_debug.log', filemode='w', level=logging.DEBUG)
    # read caffemodel weights
    net = caffe.Net(prototxt_file, input_caffemodel, caffe.TEST)
    for layer in net.params:
        pruning_percentage = pruning_percentage_dict[layer]

        # find the absolute threshold with percentile lower than pruning_percentage
        weights = net.params[layer][0].data
        threshold = np.percentile(np.abs(weights), pruning_percentage * 100)
        weights[np.abs(weights) < threshold] = 0

        biases = net.params[layer][1].data
        threshold = np.percentile(np.abs(biases), pruning_percentage * 100)
        biases[np.abs(biases) < threshold] = 0

    # finish pruning and writing weights to temporary caffemodel
    net.save(output_caffemodel)
    # logging.info('Pruning done')


def generate_best_models(dest_dir, log_file, caffemodel, prototxt):
    results = read_log(log_file)
    best_results = find_best_logs(results, accuracy_range=(0, 0.55), bin_width=0.05)
    for res in best_results:
        model_dest = os.path.join(dest_dir, 'latency_{}_acc_{}.caffemodel'.format(int(res.latency), int(res.accuracy*100)))
        prune(caffemodel, prototxt, model_dest, res.pruning_dict)
        print('Finish model', model_dest)


if __name__ == '__main__':
    with open(sys.argv[4]) as fo:
        pruning_percentage_dict = json.load(fo)
    prune(sys.argv[1], sys.argv[2], sys.argv[3], pruning_percentage_dict)

    # generate_best_models(dest_dir='results/models', log_file='results/mbo_10_1000_10_2.log',
    #                      caffemodel='models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
    #                      prototxt='models/bvlc_reference_caffenet/train_val.prototxt')

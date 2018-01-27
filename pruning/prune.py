import numpy as np
import sys
import json
import os
# supress log output from caffe loading
os.environ['GLOG_minloglevel'] = '2'
import caffe


def prune(caffemodel_file, prototxt_file, temp_caffemodel_file, pruning_percentage_dict):
    # read caffemodel weights
    net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)
    for layer in net.params:
        pruning_percentage = pruning_percentage_dict[layer]

        # find the absolute threshold with percentile lower than pruning_percentage
        weights = net.params[layer][0].data
        threshold = np.percentile(np.abs(weights), pruning_percentage * 100)
        weights[np.abs(weights) < threshold] = 0

        biases = net.params[layer][1].data
        threshold = np.percentile(np.abs(biases), pruning_percentage * 100)
        biases[np.abs(biases) < threshold] = 0

    # finish pruning and write weights to temporary caffemodel
    net.save(temp_caffemodel_file)


if __name__ == '__main__':
    with open(sys.argv[4]) as fo:
        pruning_percentage_dict = json.load(fo)
    prune(sys.argv[1], sys.argv[2], sys.argv[3], pruning_percentage_dict)

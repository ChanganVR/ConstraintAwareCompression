"""add biases to conv layers in pretrained resnet
"""

from __future__ import division
import os
import numpy as np
import re
from collections import defaultdict
os.environ['GLOG_minloglevel'] = '2'
import caffe
# caffe.set_device(0)
# caffe.set_mode_gpu()


def convert_resnet_caffemodel(input_caffemodel, prototxt, output_caffemodel, solver_file):
    # read and prune caffemodel weights
    net = caffe.Net(prototxt, input_caffemodel, caffe.TEST)
    layer_params = defaultdict(list)
    for layer in net.params:
        for blob_index in range(len(net.params[layer])):
            layer_params[layer].append(net.params[layer][blob_index].data)
        # weights = net.params[layer][0].data
        # biases = net.params[layer][1].data
    del net

    solver = caffe.get_solver(solver_file)
    # converted_net = caffe.Net(converted_prototxt, caffe.TEST)
    solver.net.forward()
    for layer in layer_params:
        for blob_index, data in enumerate(layer_params[layer]):
            np.copyto(solver.net.params[layer][blob_index].data, data)
            if re.match('res.*branch.{1,2}', layer):
                print('Layer {} non-zero weights density: {:.4f}'.format(layer, np.count_nonzero(
                    solver.net.params[layer][blob_index].data) / np.ma.size(solver.net.params[layer][blob_index].data)))
    solver.net.save(output_caffemodel)


def prune_resnet(input_caffemodel, prototxt, output_caffemodel='out_all_layer.caffemodel'):
    net = caffe.Net(prototxt, input_caffemodel, caffe.TEST)
    for layer in net.params:
        if len(net.params[layer]) >= 1:
            weights = net.params[layer][0].data
            threshold = np.percentile(np.abs(weights), 99)
            weights[np.abs(weights) < threshold] = 0
        print('Layer {} non-zero weights density: {:.4f}'.format(layer, np.count_nonzero(weights) / np.ma.size(weights)))

    net.save(output_caffemodel)


if __name__ == '__main__':
    convert_resnet_caffemodel('models/resnet/logs/0.0001_0.00005_0.0001_10_0_0_0_0_2017-04-02-22-29-02/caffenet_train_iter_2000000.caffemodel', 'models/resnet/ResNet-50-train-val.prototxt',
                              'models/resnet/pruned_converted_all_layers.caffemodel', 'models/resnet/solver.prototxt')

    prune_resnet('models/resnet/pruned_converted_all_layers.caffemodel', 'models/resnet/ResNet-50-train-val_converted_all_biases.prototxt')


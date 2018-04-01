from __future__ import division
import os
import sys
import numpy as np
from collections import defaultdict
import logging
import re


def layers_of_interest(network):
    if network == 'alexnet':
        layers = ['conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    elif network == 'resnet':
        layers = ['res2a_branch2b', 'res2b_branch2b', 'res2c_branch2b',
                  'res3a_branch2b', 'res3b_branch2b', 'res3c_branch2b', 'res3d_branch2b',
                  'res4a_branch2b', 'res4b_branch2b', 'res4c_branch2b', 'res4d_branch2b', 'res4e_branch2b', 'res4f_branch2b',
                  'res5a_branch2b', 'res5b_branch2b', 'res5c_branch2b']
    elif network == 'googlenet':
        layers = ['conv2/3x3',
                  'inception_3a/3x3', 'inception_3a/5x5', 'inception_3b/3x3', 'inception_3b/5x5',
                  'inception_4a/3x3', 'inception_4a/5x5', 'inception_4b/3x3', 'inception_4b/5x5',
                  'inception_4c/3x3', 'inception_4c/5x5', 'inception_4d/3x3', 'inception_4d/5x5',
                  'inception_4e/3x3', 'inception_4e/5x5', 'inception_5a/3x3', 'inception_5a/5x5',
                  'inception_5b/3x3', 'inception_5b/5x5']
    else:
        raise NotImplementedError
    return layers


def calculate_compression_rate(network, input_caffemodel, prototxt):
    os.environ['GLOG_minloglevel'] = '2'
    import caffe

    # read and prune caffemodel weights
    net = caffe.Net(prototxt, input_caffemodel, caffe.TEST)
    total_params = 0
    non_zeros = 0
    orig_dict = dict()
    layer_dict = dict()
    layers = layers_of_interest(network)
    for layer in layers:
        weights = net.params[layer][0].data
        biases = net.params[layer][1].data
        layer_params = np.ma.size(weights) + np.ma.size(biases)
        total_params += layer_params
        layer_non_zeros = np.count_nonzero(weights) + np.count_nonzero(biases)
        non_zeros += layer_non_zeros
        layer_dict[layer] = layer_non_zeros / layer_params
        orig_dict[layer] = layer_params

    layers = [layer for layer, _ in sorted(orig_dict.items())]
    pruning_percentages = [percent for _, percent in sorted(orig_dict.items())]
    print('Layerwise weights of caffemodel:')
    print(('{:<10}'*len(layers)).format(*layers))
    print(('{:<10}'*len(layers)).format(*pruning_percentages))
    print('Total number of weights: {}'.format(total_params))

    # print density
    layers = [layer for layer, _ in sorted(layer_dict.items())]
    pruning_percentages = [percent for _, percent in sorted(layer_dict.items())]
    print('Layerwise compression rate:')
    print(('{:<10}'*len(layers)).format(*layers))
    print(('{:<10.4f}'*len(layers)).format(*pruning_percentages))
    compression_rate = non_zeros / total_params
    print('Caffemodel non-zero density: {:4f}'.format(compression_rate))


def test_layerwise_latency(network, input_caffemodel, prototxt, test_iters):
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1'
    output_file = 'results/test_latency.txt'
    command = ['build/tools/caffe.bin', 'test', '-model', prototxt, '-weights', input_caffemodel,
               '-iterations', str(test_iters+1), '>' + output_file, '2>&1']
    logging.debug(' '.join(command))
    os.system(' '.join(command))

    constraint = re.findall(r"C_(\d+)", input_caffemodel)
    if len(constraint) == 0:
        constraint = None
    elif len(constraint) == 1:
        constraint = float(constraint[0])
    else:
        raise ValueError('More that 1 constraint')

    with open(output_file) as fo:
        text = fo.read()
        layers = layers_of_interest(network)
        # ignore the first running time due to initialization
        total_latency = [float(x) for x in re.findall(r"Total forwarding time: (\d+\.\d+) ms", text)[1:]]
        if constraint is not None:
            violation_times = sum([latency > constraint for latency in total_latency])
            print("Number of violation for {} latency constraint is {} out of {} runs".
                  format(constraint, violation_times, test_iters))
        layer_dict = defaultdict(list)
        for layer in layers:
            layer_dict[layer] = [float(x) for x in re.findall(r"Test time of {}\s*(\d+\.\d+) ms".format(layer), text)[1:]]
        print('{} runs test latency: {}'.format(test_iters, ' '.join([str(x) for x in total_latency])))
        print('Min latency: {}, max latency: {}, avg latency: {}, stdev latency: {}'.
              format(min(total_latency), max(total_latency), sum(total_latency) / len(total_latency),
                     np.std(total_latency)))
        print('Averaged layerwise latency over {} runs:'.format(test_iters))
        latencies = [sum(layer_dict[layer]) / len(layer_dict[layer]) for layer in layers]
        print(('{:<10}'*len(layers)).format(*layers))
        print(('{:<10.4f}'*len(layers)).format(*latencies))

        print('STDEV of layerwise latency over {} runs:'.format(test_iters))
        latencies = [np.std(layer_dict[layer]) for layer in layers]
        print(('{:<10}'*len(layers)).format(*layers))
        print(('{:<10.4f}'*len(layers)).format(*latencies))


if __name__ == '__main__':
    network = sys.argv[1]
    input_caffemodel = sys.argv[2]
    prototxt = sys.argv[3]
    test_layerwise_latency(network, input_caffemodel, prototxt, test_iters=20)
    calculate_compression_rate(network, input_caffemodel, prototxt)

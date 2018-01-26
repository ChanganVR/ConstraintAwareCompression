from __future__ import print_function
from __future__ import division
import sys
sys.path.append('/local-scratch/changan-home/SkimCaffe/python')
import time
import subprocess
import numpy as np
import os
import logging
# supress log output from caffe
os.environ['GLOG_minloglevel'] = '2'
import caffe

logging.basicConfig(filename='results/pruning.log', filemode='w', level=logging.INFO)


def alexnet_target_function(**pruning_percentage_dict):
    # hyper importance factor alpha and beta
    alpha = 0.1
    beta = 0.2
    test_iters = 11

    # prune the network according to the parameters
    prototxt_file = 'models/bvlc_reference_caffenet/train_val.prototxt'
    caffemodel_file = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    temp_caffemodel_file = 'results/temp_alexnet.caffemodel'
    prune(caffemodel_file, prototxt_file, temp_caffemodel_file, pruning_percentage_dict)

    # run the pruned caffemodel and get the accuracy, latency
    # recover log output from caffe
    os.environ['GLOG_minloglevel'] = '0'
    accuracy = test_accuracy(prototxt_file, temp_caffemodel_file)
    latency = test_latency(prototxt_file, temp_caffemodel_file, test_iters)

    return accuracy + alpha * latency


def test_accuracy(prototxt_file, temp_caffemodel_file):
    start = time.time()
    output_file = 'results/test_accuracy.txt'
    command = ['build/tools/caffe.bin', 'test', '-gpu', '0', '-model', prototxt_file, '-weights', temp_caffemodel_file,
               '>'+output_file, '2>&1']
    logging.debug(' '.join(command))
    os.system(' '.join(command))

    # read accuracy from output_file
    accuracy = -1
    with open(output_file) as fo:
        lines = fo.readlines()[::-1]
        # search for lines containing "accuracy ="
        for line in lines:
            if 'accuracy =' in line:
                accuracy = float(line.strip().split()[-1])
                break

    if accuracy == -1:
        logging.error('Fail to read test_accuracy.txt')
    logging.info('Testing accuracy takes {:.2f} seconds'.format(time.time() - start))
    logging.info('Accuracy: %s', accuracy)
    return accuracy


def test_latency(prototxt_file, temp_caffemodel_file, test_iters):
    start = time.time()
    output_file = 'results/test_latency.txt'
    command = ['build/tools/caffe.bin', 'test', '-model', prototxt_file, '-weights', temp_caffemodel_file,
               '-iterations', str(test_iters), '>'+output_file, '2>&1']
    logging.debug(' '.join(command))
    os.system(' '.join(command))

    # read accuracy from output_file
    with open(output_file) as fo:
        lines = fo.readlines()
        # search for lines containing "accuracy =" and skip the forwarding time
        lines = [line for line in lines if 'Total forwarding time:' in line]
        times = [float(line.strip().split()[-2]) for line in lines[1:]]

    if len(times) == 0:
        logging.error('Fail to read test_latency.txt')
    latency = sum(times) / len(times)
    logging.info('Testing latency takes {:.2f} seconds'.format(time.time() - start))
    logging.info('Latency: %s', latency)
    return latency


def prune(caffemodel_file, prototxt_file, temp_caffemodel_file, pruning_percentage_dict):
    start = time.time()
    # read caffemodel weights
    net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)
    for layer in net.params:
        pruning_percentage = pruning_percentage_dict[layer]
        logging.info('Layer {} has weights size {} and biases size {}'.
                     format(layer, np.ma.size(net.params[layer][0].data), np.ma.size(net.params[layer][1].data)))

        # find the absolute threshold with percentile lower than pruning_percentage
        weights = net.params[layer][0].data
        threshold = np.percentile(np.abs(weights), pruning_percentage * 100)
        weights[np.abs(weights) < threshold] = 0

        biases = net.params[layer][1].data
        threshold = np.percentile(np.abs(biases), pruning_percentage * 100)
        biases[np.abs(biases) < threshold] = 0

        # sparsity = 1 - (np.count_nonzero(weights) + np.count_nonzero(biases)) / (np.ma.size(weights) + np.ma.size(biases))
        # logging.info('Layer {} has sparsity {}'.format(layer, sparsity))
        logging.info('Layer {} has pruning percentage {} and pruning threshold {}'.
                     format(layer, pruning_percentage, threshold))

    # finish pruning and write weights to temporary caffemodel
    net.save(temp_caffemodel_file)
    logging.info('Temp model is written to disk')
    logging.info('Pruning takes {:.2f} seconds'.format(time.time() - start))


if __name__ == '__main__':
    loss = alexnet_target_function(conv1=0.5, conv2=0.5, conv3=0.5, conv4=0.5, conv5=0.5, fc6=0.5, fc7=0.5, fc8=0.5)
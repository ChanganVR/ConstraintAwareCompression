from __future__ import print_function
from __future__ import division
import time
import os
import logging
import json


os.environ['OMP_NUM_THREADS'] = '4'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1'

original_latency = 0


def alexnet_objective_function(**pruning_percentage_dict):
    start = time.time()
    # hyper importance factor alpha and beta
    # trade off 1 percent estimated accuracy(without training) for 50 ms final speedup (latency difference)
    if not hasattr(alexnet_objective_function, 'latency_tradeoff'):
        raise ValueError('Latency tradeoff factor for alexnet target function is not set')
    alpha = 1 / alexnet_objective_function.latency_tradeoff
    test_iters = 3

    # prune the network according to the parameters
    original_prototxt_file = 'models/bvlc_reference_caffenet/train_val.prototxt'
    sconv_prototxt_file = 'models/bvlc_reference_caffenet/test_direct_sconv_mkl.prototxt'
    caffemodel_file = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    temp_caffemodel_file = 'results/temp_alexnet.caffemodel'

    # test original caffemodel latency with sconv
    global original_latency
    if original_latency == 0:
        original_latency = test_latency(original_prototxt_file, caffemodel_file, test_iters)
        logging.info('{:<30} {}'.format('Original latency(ms):', original_latency))

    # prune and run the pruned caffemodel to get the accuracy, latency
    prune(caffemodel_file, original_prototxt_file, temp_caffemodel_file, pruning_percentage_dict)
    # batch size for latency is 8, for accuracy is 50
    # iteration number for latency is 3, for accuracy is 50
    latency = test_latency(sconv_prototxt_file, temp_caffemodel_file, test_iters)
    accuracy = test_accuracy(original_prototxt_file, temp_caffemodel_file)

    # objective is function of accuracy and latency
    logging.info('{:<30} {:.2f}'.format('Total time(s):', time.time() - start))
    objective = accuracy * 100 + alpha * (original_latency - latency)
    logging.info('{:<30} {:.2f}'.format('Objective value:', objective))

    return objective


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
    logging.info('{:<30} {:.2f}'.format('Testing accuracy takes(s):', time.time() - start))
    logging.info('{:<30} {:.2f}'.format('Accuracy:', accuracy))
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
        # discard the first running, which is usually not stable
        times = [float(line.strip().split()[-2]) for line in lines[1:]]

    if len(times) == 0:
        logging.error('Fail to read test_latency.txt')
    elif len(times) != test_iters-1:
        logging.warning('Test_latency can not find {} forwarding runs'.format(test_iters-1))
    latency = sum(times) / len(times)

    logging.info('{:<30} {:.2f}'.format('Testing latency takes(s):', time.time() - start))
    logging.info('{:<30} {:.2f}'.format('Latency(ms):', latency))
    return latency


def prune(caffemodel_file, prototxt_file, temp_caffemodel_file, pruning_percentage_dict):
    start = time.time()
    logging.info('=================================>>>Pruning starts<<<=================================')
    logging.info('conv1\t\tconv2\t\tconv3\t\tconv4\t\tconv5\t\tfc6\t\t\tfc7\t\t\tfc8')
    pruning_percentages = '{conv1} {conv2} {conv3} {conv4} {conv5} {fc6} {fc7} {fc8}'.format(**pruning_percentage_dict)
    logging.info('\t\t'.join(['%.2f' % float(x) for x in pruning_percentages.split()]))
    pruning_dict_file = 'results/pruning_dict.txt'
    with open(pruning_dict_file, 'w') as fo:
        json.dump(pruning_percentage_dict, fo)
    command = ['python', 'pruning/prune.py', caffemodel_file, prototxt_file,
               temp_caffemodel_file, pruning_dict_file]
    logging.debug(' '.join(command))
    os.system(' '.join(command))
    logging.info('{:<30} {:.2f}'.format('Pruning takes(s):', time.time() - start))


if __name__ == '__main__':
    logging.basicConfig(filename='results/pruning_debug.log', filemode='w', level=logging.DEBUG)
    # no pruning, basically copy caffemodel
    loss = alexnet_objective_function(conv1=0, conv2=0, conv3=0, conv4=0, conv5=0, fc6=0, fc7=0, fc8=0)

    # loss = alexnet_target_function(conv1=0.5, conv2=0.5, conv3=0.5, conv4=0.5, conv5=0.5, fc6=0.5, fc7=0.5, fc8=0.5)

    # skimcaffe params
    # loss = alexnet_target_function(conv1=0, conv2=0.85, conv3=0.93, conv4=0.91, conv5=0.88,
    #                                fc6=0.90, fc7=0.84, fc8=0.74)

    # clip-Q params
    # loss = alexnet_target_function(conv1=0.21, conv2=0.36, conv3=0.43, conv4=0.32, conv5=0.31,
    #                                fc6=0.96, fc7=0.95, fc8=0.74)

from __future__ import division, print_function
import os
import logging
import time
import re
import ConfigParser
from shutil import copyfile
import caffe
from objective_functions import prune, test_latency, test_env


temp_caffemodel = 'results/temp_pruned.caffemodel'
original_prototxt = None
original_caffemodel = None
test_env_prototxt = None
sconv_prototxt = None
finetune_net = None
network = None
dataset = None


def check_constraint(constraint, pruning_percentage, test_iters):
    pruning_dict = {layer: pruning_percentage for layer in ['conv2', 'conv3', 'conv4', 'conv5',
                                                            'fc6', 'fc7', 'fc8']}
    prune(network, original_caffemodel, original_prototxt, temp_caffemodel, pruning_dict)
    latency = test_latency(original_prototxt, temp_caffemodel, test_iters)
    return latency < constraint


def binary_search(original_latency, constraint):
    test_iters = 11
    left = 0.5
    right = 1
    interval = 0.001

    while not test_env(original_latency, original_caffemodel, 10000):
        logging.warning('Environment abnormal. Sleep for 3 seconds')
        time.sleep(3)

    while right - left > interval:
        mid = (left+right)/2
        satisfied = check_constraint(constraint, mid, test_iters)
        if satisfied:
            right = mid
        else:
            left = mid

    return right


def prune_and_finetune(pruning_percentage, local_config, finetune_solver, best_sampled_caffemodel, finetuning_logfile):
    pruning_dict = {layer: pruning_percentage for layer in ['conv2', 'conv3', 'conv4', 'conv5',
                                                            'fc6', 'fc7', 'fc8']}
    prune(network, original_caffemodel, original_prototxt, 'best_sampled.caffemodel', pruning_dict)
    command = ['python', 'pruning/fine_tune.py', 'best_sampled.caffemodel', finetune_net,
               best_sampled_caffemodel, local_config, finetune_solver, finetuning_logfile, dataset, network]
    os.system(' '.join(command))
    logging.debug(' '.join(command))
    if not os.path.exists(best_sampled_caffemodel):
        logging.error('Cannot find the finetuned caffemodel')

    # find acc/iter information in fine-tuning
    with open(finetuning_logfile) as fo:
        log = fo.read()
    acc_before = re.findall(r"Accuracy before: (0\.\d+)", log)[-1]
    acc_after = re.findall(r"Accuracy after: (0\.\d+)", log)[-1]
    total_iterations = re.findall(r"Total iterations: (\d+)", log)[-1]
    logging.info('Accuracy before: {}'.format(acc_before))
    logging.info('Accuracy after: {}'.format(acc_after))
    logging.info('Number of iterations: {}'.format(total_iterations))


def main():
    output_folder = 'result/unconstrained_binary_search'
    config_file = 'cfp.config'
    local_config = os.path.join(output_folder, os.path.basename(config_file))
    copyfile(config_file, local_config)
    finetune_solver = os.path.join(output_folder, 'finetune_solver.prototxt')
    best_sampled_caffemodel = os.path.join(output_folder, 'best_sampled.caffemodel')
    finetuning_logfile = os.path.join(output_folder, 'finetuning.log')

    config = ConfigParser.RawConfigParser()
    config.read(config_file)

    # input parameter
    constraint = config.getfloat('input', 'constraint')
    network = config.get('input', 'network')
    dataset = config.get('input', 'dataset')

    model_dir = 'models/bvlc_reference_caffenet'
    if dataset == 'imagenet':
        original_prototxt = os.path.join(model_dir, 'train_val.prototxt')
        original_caffemodel = os.path.join(model_dir, 'bvlc_reference_caffenet.caffemodel')
        test_env_prototxt = os.path.join(model_dir, 'test_env.prototxt')
        sconv_prototxt = os.path.join(model_dir, 'test_direct_sconv_mkl.prototxt')
        finetune_net = os.path.join(model_dir, 'train_val_ft.prototxt')
    else:
        original_prototxt = os.path.join(model_dir, 'train_val_dtd.prototxt')
        original_caffemodel = os.path.join(model_dir, 'bvlc_reference_caffenet_dtd.caffemodel')
        finetune_net = os.path.join(model_dir, 'train_val_ft_dtd.prototxt')
        test_env_prototxt = os.path.join(model_dir, 'test_env_dtd.prototxt')
        sconv_prototxt = os.path.join(model_dir, 'test_direct_sconv_mkl_dtd.prototxt')

    global original_prototxt
    global original_caffemodel
    global test_env_prototxt
    global sconv_prototxt
    global finetune_net
    global network
    global dataset

    if network == 'alexnet':
        if dataset == 'imagenet':
            # i7-4790 CPU @ 3.60GHz
            original_latency = 238
        else:
            # i7-7700 CPU @ 3.60GHz
            original_latency = 207
    else:
        assert True

    pruning_percentage = binary_search(original_latency, constraint)
    prune_and_finetune(pruning_percentage, local_config, finetune_solver, best_sampled_caffemodel, finetuning_logfile)


if __name__ == '__main__':
    main()

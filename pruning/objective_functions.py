from __future__ import print_function
from __future__ import division
import time
import os
import sys
import logging
import json
import time
import re
from utils import read_log
import numpy as np

# prune the network according to the parameters
original_prototxt = None
original_caffemodel = None
# bo acc is tested on the train set
bo_acc_prototxt = None
# test env with specific val batch size
test_env_prototxt = None
# conv mode needs to be sparse
sconv_prototxt = None
temp_caffemodel = 'results/temp_pruned.caffemodel'
test_latency_iters = 11


def matlab_objective_function(input_caffemodel, last_constraint, current_constraint, output_prefix, original_latency,
                              constraint_type, constrained_bo, tradeoff_factor, network, dataset, look_ahead):
    objective_func = objective_function
    objective_func.input_caffemodel = input_caffemodel
    objective_func.constraint = current_constraint
    objective_func.original_latency = original_latency
    objective_func.constraint_type = constraint_type
    objective_func.constrained_bo = constrained_bo
    objective_func.tradeoff_factor = tradeoff_factor
    objective_func.network = network
    objective_func.dataset = dataset
    objective_func.original_latency = original_latency
    objective_func.look_ahead = look_ahead

    global original_prototxt
    global original_caffemodel
    global bo_acc_prototxt
    global test_env_prototxt
    global sconv_prototxt
    if network == 'alexnet':
        original_prototxt = 'models/bvlc_reference_caffenet/train_val.prototxt'
        original_caffemodel = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
        model_dir = 'models/bvlc_reference_caffenet'
    elif network == 'resnet':
        original_prototxt = 'models/resnet/ResNet-50-train-val_converted.prototxt'
        original_caffemodel = 'models/resnet/ResNet-50-model_converted.caffemodel'
        model_dir = 'models/resnet'
    elif network == 'googlenet':
        original_prototxt = 'models/bvlc_googlenet/train_val.prototxt'
        original_caffemodel = 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
        model_dir = 'models/bvlc_googlenet'
    else:
        raise NotImplementedError

    if dataset == 'imagenet':
        bo_acc_prototxt = os.path.join(model_dir, 'bo_acc.prototxt')
        test_env_prototxt = os.path.join(model_dir, 'test_env.prototxt')
        sconv_prototxt = os.path.join(model_dir, 'test_direct_sconv_mkl.prototxt')
    else:
        original_prototxt = original_prototxt.replace('.prototxt', '_dtd.prototxt')
        original_caffemodel = original_caffemodel.replace('.caffemodel', '_dtd.caffemodel')
        bo_acc_prototxt = os.path.join(model_dir, 'bo_acc_dtd.prototxt')
        test_env_prototxt = os.path.join(model_dir, 'test_env_dtd.prototxt')
        sconv_prototxt = os.path.join(model_dir, 'test_direct_sconv_mkl_dtd.prototxt')

    # configure output log
    log_file = output_prefix + 'bo.log'
    if not hasattr(objective_func, 'log_file') or objective_func.log_file != log_file:
        reload(logging)
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO,
                            format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        logging.info('Constraint type: {}'.format(constraint_type))
        logging.info('Input caffemodel: {}'.format(input_caffemodel))
        logging.info('Input network: {}'.format(network))
        if constrained_bo:
            logging.info('Running constrained bayesian optimization')
            logging.info('Last constraint: {:.2f}'.format(last_constraint))
            logging.info('Current constraint: {:.2f}'.format(current_constraint))
        else:
            logging.info('Running unconstrained bayesian optimization with tradeoff factor {}'.format(tradeoff_factor))
        objective_func.log_file = log_file
        if constrained_bo and constraint_type == 'latency':
            while not test_env(original_latency, input_caffemodel, last_constraint):
                logging.warning('Environment abnormal. Sleep for 3 seconds')
                time.sleep(3)
        else:
            os.environ['LD_LIBRARY_PATH'] = '/local-scratch/changan-home/lib/boost/lib:/local-scratch/changan-home/intel/itac/2018.1.017/intel64/slib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/../tbb/lib/intel64_lin/gcc4.4:/usr/local-linux/lib'

    return objective_func


def objective_function(**pruning_dict):
    start = time.time()
    # set some hyper parameters
    constraint = objective_function.constraint
    input_caffemodel = objective_function.input_caffemodel
    constraint_type = objective_function.constraint_type
    constrained_bo = objective_function.constrained_bo
    tradeoff_factor = objective_function.tradeoff_factor
    network = objective_function.network
    dataset = objective_function.dataset
    original_latency = objective_function.original_latency
    look_ahead = objective_function.look_ahead

    if network == 'alexnet':
        if dataset == 'imagenet':
            test_acc_iters = 12
        elif dataset == 'dtd':
            test_acc_iters = 10
    elif network == 'resnet':
        if dataset == 'imagenet':
            test_acc_iters = 100
    elif network == 'googlenet':
        if dataset == 'imagenet':
            test_acc_iters = 30
    else:
        raise NotImplementedError

    if constraint_type == 'latency':
        prune(network, input_caffemodel, original_prototxt, temp_caffemodel, pruning_dict)
        latency = test_latency(sconv_prototxt, temp_caffemodel, test_latency_iters)
        constraint_violation = latency - constraint
        if look_ahead:
            assert dataset == 'dtd'
        accuracy = test_accuracy(bo_acc_prototxt, temp_caffemodel, test_acc_iters, look_ahead, network)
        if constrained_bo:
            objective = -1 * accuracy * 100
        else:
            objective = -1 * (accuracy * 100 + tradeoff_factor * (original_latency - latency))
    elif constraint_type == 'compression_rate':
        assert dataset == 'imagenet'
        compression_rate, accuracy = prune_and_test(network, input_caffemodel, bo_acc_prototxt,
                                                    test_acc_iters, pruning_dict)
        constraint_violation = compression_rate - constraint
        if constrained_bo:
            objective = -1 * accuracy * 100
        else:
            objective = -1 * (accuracy * 100 + tradeoff_factor * (1 - compression_rate) * 100)
    else:
        raise NotImplemented

    logging.debug('{:<30} {:.2f}'.format('Total time(s):', time.time() - start))
    logging.info('{:<30} {:.4f}'.format('Objective value:', objective))
    if constrained_bo:
        return objective, constraint_violation
    else:
        return objective


def convert_pruning_dict(network, pruning_dict):
    if network == 'alexnet':
        converted_pruning_dict = pruning_dict
    elif network == 'resnet':
        resnet_blocks = dict()
        resnet_blocks['conv2'] = ['res2a_branch2b', 'res2b_branch2b', 'res2c_branch2b']
        resnet_blocks['conv3'] = ['res3a_branch2b', 'res3b_branch2b', 'res3c_branch2b', 'res3d_branch2b']
        resnet_blocks['conv4'] = ['res4a_branch2b', 'res4b_branch2b', 'res4c_branch2b', 'res4d_branch2b',
                                  'res4e_branch2b', 'res4f_branch2b']
        resnet_blocks['conv5'] = ['res5a_branch2b', 'res5b_branch2b', 'res5c_branch2b']
        converted_pruning_dict = dict()
        for block in resnet_blocks:
            for layer in resnet_blocks[block]:
                converted_pruning_dict[layer] = pruning_dict[block]
    elif network == 'googlenet':
        inception_blocks = dict()
        inception_blocks['conv2'] = ['conv2/3x3']
        inception_blocks['i3a'] = ['inception_3a/3x3', 'inception_3a/5x5']
        inception_blocks['i3b'] = ['inception_3b/3x3', 'inception_3b/5x5']
        inception_blocks['i4a'] = ['inception_4a/3x3', 'inception_4a/5x5']
        inception_blocks['i4b'] = ['inception_4b/3x3', 'inception_4b/5x5']
        inception_blocks['i4c'] = ['inception_4c/3x3', 'inception_4c/5x5']
        inception_blocks['i4d'] = ['inception_4d/3x3', 'inception_4d/5x5']
        inception_blocks['i4e'] = ['inception_4e/3x3', 'inception_4e/5x5']
        inception_blocks['i5a'] = ['inception_5a/3x3', 'inception_5a/5x5']
        inception_blocks['i5b'] = ['inception_5b/3x3', 'inception_5b/5x5']
        converted_pruning_dict = dict()
        for block in inception_blocks:
            for layer in inception_blocks[block]:
                converted_pruning_dict[layer] = pruning_dict[block]
    else:
        raise NotImplementedError

    layers = [layer for layer, _ in sorted(converted_pruning_dict.items())]
    pruning_percentages = [percent for _, percent in sorted(converted_pruning_dict.items())]
    logging.debug(('{:<10}'*len(layers)).format(*layers))
    logging.debug(('{:<10.4f}'*len(layers)).format(*pruning_percentages))

    return converted_pruning_dict


def prune_and_test(network, input_caffemodel, prototxt, test_acc_iters, pruning_dict):
    # prune the input caffemodel, calculate its compression rate and accuracy on training set
    start = time.time()
    logging.info('=================================>>>Pruning starts<<<=================================')
    layers = [layer for layer, _ in sorted(pruning_dict.items())]
    pruning_percentages = [percent for _, percent in sorted(pruning_dict.items())]
    logging.info(('{:<10}'*len(layers)).format(*layers))
    logging.info(('{:<10.4f}'*len(layers)).format(*pruning_percentages))

    pruning_dict = convert_pruning_dict(network, pruning_dict)
    pruning_dict_file = 'results/pruning_dict.txt'
    with open(pruning_dict_file, 'w') as fo:
        json.dump(pruning_dict, fo)
    command = ['python', 'pruning/prune_and_test.py', input_caffemodel, prototxt, str(test_acc_iters), pruning_dict_file]
    logging.debug(' '.join(command))
    ret = os.system(' '.join(command))
    if ret == 0:
        logging.debug('{:<30} {:.2f}'.format('Pruning takes(s):', time.time() - start))
    else:
        logging.error('Fail to prune the caffemodel')

    with open('results/prune_and_test.txt') as fo:
        compression_rate, accuracy = [float(x) for x in fo.read().strip().split()]
        logging.info('{:<30} {:.4f}'.format('Compression rate:', compression_rate))
        logging.info('{:<30} {:.4f}'.format('Accuracy:', accuracy))
    return compression_rate, accuracy


def test_env(original_latency, input_caffemodel, last_constraint):
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1'
    os.environ['LD_LIBRARY_PATH'] = '/local-scratch/changan-home/lib/boost/lib:/local-scratch/changan-home/intel/itac/2018.1.017/intel64/slib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/lib/boost/lib:/local-scratch/changan-home/intel/itac/2018.1.017/intel64/slib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/../tbb/lib/intel64_lin/gcc4.4:/usr/local-linux/lib'
    # test original caffemodel latency in current environment, should not have a big different with normal latency
    if not hasattr(objective_function, 'test_env'):
        logging.info('=================================>>>Test environment<<<=================================')
        logging.info('{:<30} {}'.format('Original latency(ms):', original_latency))

        logging.info('Test original caffemodel latency with normal conv:')
        test_env_latency = test_latency(test_env_prototxt, original_caffemodel, test_latency_iters)
        if test_env_latency - original_latency > 3:
            logging.error('Test original latency is off from normal latency too much. Check the environment!')
            return False

        logging.info('Test input caffemodel latency with sparse conv:')
        test_input_latency = test_latency(sconv_prototxt, input_caffemodel, test_latency_iters)
        if test_input_latency - last_constraint > 3:
            logging.error('Test input latency is off from last constraint too much. Check the environment!')
            return False

        objective_function.test_env = True
        return True


def test_accuracy(prototxt_file, temp_caffemodel_file, iterations, look_ahead, network):
    start = time.time()
    output_file = 'results/test_accuracy.txt'
    if look_ahead:
        solver_file = 'models/bvlc_reference_caffenet/lookahead_ft_dtd.prototxt'
        command = ['build/tools/caffe.bin', 'train', '-gpu', '0', '-solver', solver_file,
                   '-weights', temp_caffemodel_file, '>'+output_file, '2>&1']
        logging.debug(' '.join(command))
        os.system(' '.join(command))
        temp_caffemodel_file = 'results/lookahead_iter_15.caffemodel'

    # test the accuracy
    command = ['build/tools/caffe.bin', 'test', '-gpu', '0', '-model', prototxt_file,
               '-weights', temp_caffemodel_file, '-iterations', str(iterations), '>'+output_file, '2>&1']
    logging.debug(' '.join(command))
    os.system(' '.join(command))

    # read accuracy from output_file
    with open(output_file) as fo:
        if network == 'alexnet':
            pattern = r"accuracy = (0\.?\d*)"
        elif network == 'resnet':
            pattern = r"top-1 = (0\.?\d*)"
        elif network == 'googlenet':
            pattern = r"loss3/top-1 = (0\.?\d*)"
        text = fo.read()
        accuracy = re.findall(pattern, text)
        if len(accuracy) == 0:
            logging.warning('Fail to read test_accuracy.txt')
        else:
            accuracy = float(accuracy[0])
    logging.debug('{:<30} {:.2f}'.format('Testing accuracy takes(s):', time.time() - start))
    logging.info('{:<30} {:.4f}'.format('Accuracy:', accuracy))
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
        text = fo.read()
        forwarding_time = re.findall(r"Total forwarding time: (\d+\.?\d+)", text)
        if len(forwarding_time) != test_iters:
            logging.error('Fail to read test_latency.txt')
        else:
            forwarding_time = [float(x) for x in forwarding_time[1:]]

    # enforce hard constraint, pick the maximum latency
    logging.debug('{} runs latency measurements: {}'.format(test_iters-1, ' '.join([str(x) for x in forwarding_time])))
    # latency = max(times)
    latency = sum(forwarding_time) / len(forwarding_time)

    logging.debug('{:<30} {:.2f}'.format('Testing latency takes(s):', time.time() - start))
    logging.info('{:<30} {:.2f}'.format('Latency(ms):', latency))
    return latency


def prune(network, caffemodel_file, prototxt_file, temp_caffemodel_file, pruning_dict):
    start = time.time()
    logging.info('=================================>>>Pruning starts<<<=================================')
    layers = [layer for layer, _ in sorted(pruning_dict.items())]
    pruning_percentages = [percent for _, percent in sorted(pruning_dict.items())]
    logging.info(('{:<10}'*len(layers)).format(*layers))
    logging.info(('{:<10.4f}'*len(layers)).format(*pruning_percentages))

    pruning_dict = convert_pruning_dict(network, pruning_dict)
    pruning_dict_file = 'results/pruning_dict.txt'
    with open(pruning_dict_file, 'w') as fo:
        json.dump(pruning_dict, fo)
    command = ['python', 'pruning/prune.py', caffemodel_file, prototxt_file, temp_caffemodel_file, pruning_dict_file]
    logging.debug(' '.join(command))
    ret = os.system(' '.join(command))
    if ret == 0:
        logging.debug('{:<30} {:.2f}'.format('Pruning takes(s):', time.time() - start))
    else:
        logging.error('Fail to prune the caffemodel')


if __name__ == '__main__':
    pass

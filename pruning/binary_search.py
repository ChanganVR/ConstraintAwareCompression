from __future__ import division, print_function
import os
import logging
import time
import re
import ConfigParser
from shutil import copyfile
import caffe
from objective_functions import prune, test_latency


temp_caffemodel = 'results/temp_pruned.caffemodel'
test_latency_iters = 11

original_prototxt = None
original_caffemodel = None
test_env_prototxt = None
sconv_prototxt = None
finetune_net = None
network = None
dataset = None


def check_constraint(constraint, pruning_percentage):
    pruning_dict = {layer: pruning_percentage for layer in ['conv2', 'conv3', 'conv4', 'conv5',
                                                            'fc6', 'fc7', 'fc8']}
    prune(network, original_caffemodel, original_prototxt, temp_caffemodel, pruning_dict)
    latency = test_latency(original_prototxt, temp_caffemodel, test_latency_iters)
    return latency < constraint


def binary_search(original_latency, constraint):
    left = 0.5
    right = 1
    interval = 0.001

    while right - left > interval:
        mid = (left+right)/2
        satisfied = check_constraint(constraint, mid)
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
    output_folder = 'results/unconstrained_binary_search'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    config_file = 'cfp.config'
    local_config = os.path.join(output_folder, os.path.basename(config_file))
    copyfile(config_file, local_config)
    finetune_solver = os.path.join(output_folder, 'finetune_solver.prototxt')
    best_sampled_caffemodel = os.path.join(output_folder, 'best_sampled.caffemodel')
    finetuning_logfile = os.path.join(output_folder, 'finetuning.log')

    global original_prototxt
    global original_caffemodel
    global test_env_prototxt
    global sconv_prototxt
    global finetune_net
    global network
    global dataset

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

    if network == 'alexnet':
        if dataset == 'imagenet':
            # i7-4790 CPU @ 3.60GHz
            original_latency = 238
        else:
            # i7-7700 CPU @ 3.60GHz
            original_latency = 240
    else:
        assert True

    while not test_env(original_latency, original_caffemodel, 10000):
        logging.warning('Environment abnormal. Sleep for 3 seconds')
        time.sleep(3)

    pruning_percentage = binary_search(original_latency, constraint)
    prune_and_finetune(pruning_percentage, local_config, finetune_solver, best_sampled_caffemodel, finetuning_logfile)


def test_env(original_latency, input_caffemodel, last_constraint):
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1'
    os.environ['LD_LIBRARY_PATH'] = '/local-scratch/changan-home/lib/boost/lib:/local-scratch/changan-home/intel/itac/2018.1.017/intel64/slib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/lib/boost/lib:/local-scratch/changan-home/intel/itac/2018.1.017/intel64/slib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/../tbb/lib/intel64_lin/gcc4.4:/usr/local-linux/lib'

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

    return True


if __name__ == '__main__':
    main()

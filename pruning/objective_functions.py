from __future__ import print_function
from __future__ import division
import time
import os
import logging
import json
import sys


def matlab_alexnet_objective_function(input_caffemodel, latency_constraint, output_prefix, original_latency):
    objective_function = alexnet_objective_function
    objective_function.input_caffemodel = input_caffemodel
    objective_function.latency_constraint = latency_constraint
    objective_function.original_latency = original_latency
    objective_function.constrained_optimization = True

    # configure output log
    log_file = output_prefix + 'bo.log'
    if not hasattr(objective_function, 'log_file') or objective_function.log_file != log_file:
        reload(logging)
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO,
                            format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        logging.info('Latency constraint: {}'.format(latency_constraint))
        objective_function.log_file = log_file

    return objective_function


def alexnet_objective_function(**pruning_percentage_dict):
    start = time.time()
    # set some hyper parameters
    if hasattr(alexnet_objective_function, 'constrained_optimization'):
        latency_constraint = alexnet_objective_function.latency_constraint
        constrained_optimization = True
    else:
        if not hasattr(alexnet_objective_function, 'latency_tradeoff'):
            raise ValueError('Latency tradeoff factor is not set')
        latency_tradeoff = alexnet_objective_function.latency_tradeoff
    if not hasattr(alexnet_objective_function, 'original_latency'):
        raise ValueError('Original latency is not set')
    original_latency = alexnet_objective_function.original_latency
    if not hasattr(alexnet_objective_function, 'input_caffemodel'):
        raise ValueError('Input caffemodel is not set')
    input_caffemodel = alexnet_objective_function.input_caffemodel
    test_iters = 3

    # prune the network according to the parameters
    original_prototxt = 'models/bvlc_reference_caffenet/train_val.prototxt'
    original_caffemodel = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    test_env_prototxt = 'models/bvlc_reference_caffenet/test_env.prototxt'
    sconv_prototxt = 'models/bvlc_reference_caffenet/test_direct_sconv_mkl.prototxt'
    temp_caffemodel = 'results/temp_alexnet.caffemodel'

    # prune and run the pruned caffemodel to get the accuracy, latency
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1'
    os.environ['LD_LIBRARY_PATH'] = '/local-scratch/changan-home/lib/boost/lib:/local-scratch/changan-home/intel/itac/2018.1.017/intel64/slib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/lib/boost/lib:/local-scratch/changan-home/intel/itac/2018.1.017/intel64/slib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mpi/mic/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.7:/local-scratch/changan-home/intel/debugger_2018/iga/lib:/local-scratch/changan-home/intel/debugger_2018/libipt/intel64/lib:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/lib/intel64_lin:/local-scratch/changan-home/intel/compilers_and_libraries_2018.1.163/linux/daal/../tbb/lib/intel64_lin/gcc4.4:/usr/local-linux/lib'
    # test original caffemodel latency in current environment, should not have a big different with normal latency
    if not hasattr(alexnet_objective_function, 'test_env'):
        logging.info('{:<30} {}'.format('Original latency(ms):', original_latency))
        test_env_latency = test_latency(test_env_prototxt, original_caffemodel, test_iters)
        logging.info('Test original latency: {}'.format(test_env_latency))
        if abs(test_env_latency - original_latency) > 10:
            logging.error('Test original latency is off from normal latency too much. Check the environment!')
        else:
            alexnet_objective_function.test_env = True
    prune(input_caffemodel, original_prototxt, temp_caffemodel, pruning_percentage_dict)
    # batch size for latency is 8, for accuracy is 50
    # iteration number for latency is 3, for accuracy is 50
    latency = test_latency(sconv_prototxt, temp_caffemodel, test_iters)
    accuracy = test_accuracy(original_prototxt, temp_caffemodel)

    # objective is function of accuracy and latency
    logging.info('{:<30} {:.2f}'.format('Total time(s):', time.time() - start))
    if constrained_optimization:
        # the bayesian optimization function in matlab minimizes the objective function
        objective = -1 * accuracy * 100
        constraint_violation = latency - latency_constraint
        logging.info('{:<30} {:.2f}'.format('Objective value:', objective))
        return objective, constraint_violation
    else:
        objective = accuracy * 100 + latency_tradeoff * (original_latency - latency)
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
    command = ['python', 'pruning/prune.py', caffemodel_file, prototxt_file, temp_caffemodel_file, pruning_dict_file]
    logging.debug(' '.join(command))
    ret = os.system(' '.join(command))
    if ret == 0:
        logging.info('{:<30} {:.2f}'.format('Pruning takes(s):', time.time() - start))
    else:
        logging.error('Fail to prune the caffemodel')


if __name__ == '__main__':
    os.chdir('/local-scratch/changan-home/SkimCaffe')
    logging.basicConfig(filename='results/objective_function_debug.log', filemode='w', level=logging.DEBUG,
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    # eng = matlab.engine.start_matlab()
    # no pruning, basically copy caffemodel
    # loss = alexnet_objective_function(conv1=0, conv2=0, conv3=0, conv4=0, conv5=0, fc6=0, fc7=0, fc8=0)

    # if len(sys.argv) != 4:
    #     raise ValueError('Input argument incorrect')
    # input_caffemodel = sys.argv[1]
    # latency_constraint = float(sys.argv[2])
    # pruning_file = sys.argv[3]
    #
    # with open(pruning_file) as fo:
    #     pruning_dict = json.load(fo)
    #
    # objective_func = matlab_alexnet_objective_function(input_caffemodel=input_caffemodel, latency_constraint=latency_constraint)
    # objective, satisfied = objective_func(**pruning_dict)
    # print(objective, satisfied)

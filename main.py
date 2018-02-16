from __future__ import division
from __future__ import print_function
import logging
import os
import time
import json
import sys
import math
import ConfigParser
from shutil import copyfile
from pruning.objective_functions import alexnet_objective_function
from pruning.bayesian_optimization import bayesian_optimization, constrained_bayesian_optimization
from pruning.utils import find_next_phase, read_log


def relaxed_constraint(iteration, relaxation_func):
    if relaxation_func == 'linear':
        return original_latency + (iteration+1)/fine_pruning_iterations * (latency_constraint - original_latency)
    elif relaxation_func == 'exponential':
        # using Newton's Law of Cooling
        # plot: 80+(238-80)*exp(-0.5x)+(80-238)*exp(-2.5) from 1 to 5
        return latency_constraint + (original_latency - latency_constraint) * math.exp(-1 * exp_factor * (iteration + 1)) \
               + (latency_constraint - original_latency) * math.exp(-1 * exp_factor * fine_pruning_iterations)
    elif relaxation_func == 'one-step':
        return latency_constraint
    else:
        raise NotImplementedError


if len(sys.argv) == 1:
    resume_training = False
elif sys.argv[1] == 'resume':
    resume_training = True
else:
    raise ValueError('Command line argument incorrect')

config_file = 'cfp.config'
config = ConfigParser.RawConfigParser()
config.read(config_file)

# fixed hyper parameters
num_threads = 4
batch_size = 32
original_latency = 238
init_points = 20
kappa = 10
constrained_optimization = True

# input parameter
latency_constraint = config.getfloat('input', 'latency_constraint')

# constrained bayesian optimization
fine_pruning_iterations = config.getint('cbo', 'fine_pruning_iterations')
exp_factor = config.getfloat('cbo', 'exp_factor')
bo_iters = config.getint('cbo', 'bo_iters')
relaxation_function = config.get('cbo', 'relaxation_function')

# fine-tuning
min_acc = config.getfloat('fine-tuning', 'min_acc')
max_iter = config.getint('fine-tuning', 'max_iter')
learning_rate = config.getfloat('fine-tuning', 'learning_rate')
gamma = config.getfloat('fine-tuning', 'gamma')
stepsize = config.getint('fine-tuning', 'stepsize')
regularization = config.get('fine-tuning', 'regularization')
weight_decay = config.getfloat('fine-tuning', 'weight_decay')

# some path variables
original_prototxt = 'models/bvlc_reference_caffenet/train_val.prototxt'
original_caffemodel = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = "models/bvlc_reference_caffenet/train_val_ft.prototxt"
output_folder = 'results/C_{}_cfp_{}_bo_{}_exp_{}_R_{}'.format(latency_constraint, fine_pruning_iterations, bo_iters,
                                                               exp_factor, relaxation_function)
finetune_solver = os.path.join(output_folder, 'finetune_solver.prototxt')
best_sampled_caffemodel = os.path.join(output_folder, 'best_sampled.caffemodel')
last_finetuned_caffemodel = os.path.join(output_folder, '0th_finetuned.caffemodel')
log_file = os.path.join(output_folder, 'fine_pruning.log')


if resume_training:
    logging.basicConfig(filename=log_file, filemode='a+', level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    t, next_phase = find_next_phase(log_file)
    logging.info('Resume training: current fine-pruning iteration is {}, next phase is {}'.format(t, next_phase))
    if next_phase == 'bayesian optimization':
        last_relaxed_constraint = relaxed_constraint(t - 1, relaxation_function)
    else:
        last_relaxed_constraint = relaxed_constraint(t, relaxation_function)
    last_finetuned_caffemodel = os.path.join(output_folder, '{}th_finetuned.caffemodel'.format(t-1))
elif os.path.exists(output_folder):
    raise IOError('{} already exist.'.format(output_folder))
else:
    os.mkdir(output_folder)
    logging.basicConfig(filename=log_file, filemode='a+', level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    logging.info('{:<40} {}'.format('Original latency:', original_latency))
    logging.info('{:<40} {}'.format('Latency constraint:', latency_constraint))
    logging.info('{:<40} {}'.format('Constrained fine-pruning iterations:', fine_pruning_iterations))
    logging.info('{:<40} {}'.format('Bayesian optimization iterations:', bo_iters))
    logging.info('{:<40} {}'.format('Relaxation function:', relaxation_function))
    logging.info('{:<40} {}'.format('Exponential cooling factor:', exp_factor))

    t = 0
    next_phase = None
    last_relaxed_constraint = 10000

    # copy current config file and create new solver
    copyfile(config_file, os.path.join(output_folder, os.path.basename(config_file)))
    with open(finetune_solver, 'w') as fo:
        fo.write('net: "{}"\n'.format(net))
        fo.write('test_iter: {}\n'.format(100))
        fo.write('test_interval: {}\n'.format(10000))
        fo.write('base_lr: {}\n'.format(learning_rate))
        fo.write('gamma: {}\n'.format(gamma))
        fo.write('lr_policy: "{}"\n'.format('step'))
        fo.write('stepsize: {}\n'.format(stepsize))
        fo.write('display: {}\n'.format(200))
        fo.write('max_iter: {}\n'.format(max_iter))
        fo.write('momentum: {}\n'.format(0.9))
        if regularization == 'L1':
            fo.write('regularization_type: {}\n'.format(regularization))
            fo.write('weight_decay: {}\n'.format(weight_decay))
        fo.write('solver_mode: {}\n'.format('GPU'))

while t < fine_pruning_iterations:
    if t == 0:
        input_caffemodel = original_caffemodel
    else:
        input_caffemodel = last_finetuned_caffemodel
    # compute relaxed constraints
    current_relaxed_constraint = relaxed_constraint(t, relaxation_function)

    if next_phase is None or next_phase == 'bayesian optimization':
        logging.info('The relaxed constraint in {}th iteration is {:.2f}'.format(t, current_relaxed_constraint))
        logging.info('Start {}th fine-pruning iteration'.format(t))
        # first do bayesian optimization given latency tradeoff factor
        start = time.time()
        if constrained_optimization:
            output_prefix = output_folder + '/' + str(t)
            constrained_bayesian_optimization(n_iter=bo_iters, init_points=init_points,
                                              input_caffemodel=input_caffemodel,
                                              last_constraint=last_relaxed_constraint,
                                              latency_constraint=current_relaxed_constraint,
                                              output_prefix=output_prefix, original_latency=original_latency)
            last_relaxed_constraint = current_relaxed_constraint
        else:
            # allow 4 percent drop in accuracy to trade off for 140 ms speedup
            # latency tradeoff function changes according to cooling function
            latency_tradeoff = (0.57-min_acc) * 100 / (last_relaxed_constraint - current_relaxed_constraint)
            objective_function = alexnet_objective_function
            objective_function.latency_tradeoff = latency_tradeoff
            objective_function.original_latency = last_relaxed_constraint
            last_relaxed_constraint = current_relaxed_constraint
            objective_function.input_caffemodel = input_caffemodel
            bayesian_optimization(n_iter=bo_iters, tradeoff_factors=(latency_tradeoff,),
                                  objective_function=objective_function, init_points=init_points, kappa=kappa)
        logging.info('Bayesian optimization in {}th iteration takes {:.2f}s'.format(t, time.time()-start))
        next_phase = None

    if next_phase is None or next_phase == 'pruning':
        # find the best point satisfying the relaxed constraints
        logs, _ = read_log(log_file=os.path.join(output_folder, str(t) + 'bo.log'))
        max_acc = 0
        max_log = None
        for log in logs:
            if log.latency <= current_relaxed_constraint and log.accuracy > max_acc:
                max_log = log
                max_acc = log.accuracy
        logging.info('The best point chosen satisfying the constraint:')
        logging.info(max_log)

        # prune best point in sampled results
        start = time.time()
        pruning_dict_file = 'results/pruning_dict.txt'
        with open(pruning_dict_file, 'w') as fo:
            json.dump(max_log.pruning_dict, fo)
        command = ['python', 'pruning/prune.py', input_caffemodel, original_prototxt,
                   best_sampled_caffemodel, pruning_dict_file]
        os.system(' '.join(command))
        if not os.path.exists(best_sampled_caffemodel):
            logging.error('Cannot find the best sampled model')
        logging.info('Pruning the best sampled model in {}th iteration takes {:.2f}s'.format(t, time.time()-start))
        next_phase = None

    if next_phase is None or next_phase == 'finetuning':
        # avoid affecting latency measurement, run fine-tuning and pruning from command line
        # fine-tune the pruned caffemodel until acc > min_acc or iteration > max_iter
        # TODO: should min_acc be a function of time? since the accuracy is harder to recover later
        start = time.time()
        last_finetuned_caffemodel = os.path.join(output_folder, '{}th_finetuned.caffemodel'.format(t))
        finetuning_logfile = last_finetuned_caffemodel.replace('caffemodel', 'log')
        command = ['python', 'pruning/fine_tune.py', best_sampled_caffemodel, finetune_solver,
                   last_finetuned_caffemodel, str(min_acc), str(max_iter), finetuning_logfile]
        os.system(' '.join(command))
        logging.debug(' '.join(command))
        if not os.path.exists(last_finetuned_caffemodel):
            logging.error('Cannot find the finetuned caffemodel')
        # find acc/iter information in fine-tuning
#        loss_pattern = r"Training iteration (?P<iter_num>\d+), loss: (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
#        acc_before =  re.findall(loss_pattern, finetuning_logfile)
#            loss_iterations.append(int(r[0]))
#            losses.append(float(r[1]))
#
        logging.info('Fine-tuning in {}th iteration takes {:.2f}s'.format(t, time.time()-start))
        next_phase = None

    t += 1




from __future__ import division
from __future__ import print_function
import logging
import os
import time
import json
import sys
import re
import math
import ConfigParser
import matlab.engine
from shutil import copyfile
from pruning.utils import find_next_phase, read_log


def relaxed_constraint(iteration, relaxation_func):
    if relaxation_func == 'linear':
        return original_latency + (iteration+1)/fine_pruning_iterations * (constraint - original_latency)
    elif relaxation_func == 'exponential':
        # using Newton's Law of Cooling
        if constraint_type == 'latency':
            # plot: 80+(238-80)*exp(-0.5x)+(80-238)*exp(-2.5) from 1 to 5
            return constraint + (original_latency - constraint) * math.exp(-1 * exp_factor * (iteration + 1)) \
               + (constraint - original_latency) * math.exp(-1 * exp_factor * fine_pruning_iterations)
        elif constraint_type == 'compression_rate':
            # plot: 0.05+(1-0.05)*exp(-0.5x)+(0.05-1)*exp(-2.5) from 1 to 5
            return constraint + (1 - constraint) * math.exp(-1 * exp_factor * (iteration + 1)) \
               + (constraint - 1) * math.exp(-1 * exp_factor * fine_pruning_iterations)
        else:
            raise NotImplementedError

    elif relaxation_func == 'one-step':
        return constraint
    else:
        raise NotImplementedError


if len(sys.argv) == 1:
    resume_training = False
elif len(sys.argv) == 3 and sys.argv[1] == 'resume':
    resume_training = True
    resume_folder = sys.argv[2]
else:
    raise ValueError('Command line argument incorrect')

if resume_training:
    config_file = os.path.join(resume_folder, 'cfp.config')
else:
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
constraint_type = config.get('input', 'constraint_type')
constraint = config.getfloat('input', 'constraint')

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
if resume_training:
    output_folder = resume_folder
else:
    if relaxation_function != 'exponential':
        output_folder = 'results/C_{:.0f}_cfp_{}_bo_{}_R_{}'.format(constraint, fine_pruning_iterations, bo_iters,
                                                                    relaxation_function)
    else:
        output_folder = 'results/C_{:.0f}_cfp_{}_bo_{}_R_{}_exp_{}'.format(constraint, fine_pruning_iterations, bo_iters,
                                                                           relaxation_function, exp_factor)
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
        last_constraint = relaxed_constraint(t - 1, relaxation_function)
    else:
        last_constraint = relaxed_constraint(t, relaxation_function)
    last_finetuned_caffemodel = os.path.join(output_folder, '{}th_finetuned.caffemodel'.format(t-1))
elif os.path.exists(output_folder):
    raise IOError('{} already exist.'.format(output_folder))
else:
    os.mkdir(output_folder)
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    # logging.info('{:<40} {}'.format('Original latency:', original_latency))
    # logging.info('{:<40} {}'.format('Latency constraint:', latency_constraint))
    # logging.info('{:<40} {}'.format('Constrained fine-pruning iterations:', fine_pruning_iterations))
    # logging.info('{:<40} {}'.format('Bayesian optimization iterations:', bo_iters))
    # logging.info('{:<40} {}'.format('Relaxation function:', relaxation_function))
    # logging.info('{:<40} {}'.format('Exponential cooling factor:', exp_factor))

    t = 0
    next_phase = None
    last_constraint = 10000

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
    current_constraint = relaxed_constraint(t, relaxation_function)

    if next_phase is None or next_phase == 'bayesian optimization':
        logging.info('The relaxed constraint in {}th iteration is {:.2f}'.format(t, current_constraint))
        logging.info('Start {}th fine-pruning iteration'.format(t))
        start = time.time()
        output_prefix = output_folder + '/' + str(t)

        # call matlab bayesian optimization code
        eng = matlab.engine.start_matlab()
        eng.addpath('/local-scratch/changan-home/SkimCaffe/pruning')
        eng.bayesian_optimization(bo_iters, init_points, input_caffemodel, last_constraint, current_constraint,
                                  output_prefix, original_latency, constraint_type)
        eng.quit()

        last_constraint = current_constraint
        logging.info('Bayesian optimization in {}th iteration takes {:.2f}s'.format(t, time.time()-start))
        next_phase = None

    if next_phase is None or next_phase == 'pruning':
        # find the best point satisfying the relaxed constraints
        logs, _ = read_log(log_file=os.path.join(output_folder, str(t) + 'bo.log'))
        min_obj = 0
        min_log = None
        for log in logs:
            if constraint_type == 'latency':
                if log.latency <= current_constraint and log.objective_value < min_obj:
                    min_obj = log.objective_value
                    min_obj = log
            elif constraint_type == 'compression_rate':
                if log.compression_rate <= current_constraint and log.objective_value < min_obj:
                    min_obj = log.objective_value
                    min_obj = log
        if min_log is None:
            logging.error('No point found satisfying the constraint')
        else:
            logging.info('The best point chosen satisfying the constraint:')
            logging.info(min_log)

        # prune best point in sampled results
        start = time.time()
        pruning_dict_file = 'results/pruning_dict.txt'
        with open(pruning_dict_file, 'w') as fo:
            json.dump(min_log.pruning_dict, fo)
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
        with open(finetuning_logfile) as fo:
            log = fo.read()
        acc_before = re.findall(r"Accuracy before: (0\.\d+)", log)[0]
        acc_after = re.findall(r"Accuracy after: (0\.\d+)", log)[0]
        total_iterations = re.findall(r"Total iterations: (\d+)", log)[0]
        logging.info('Accuracy before: {}'.format(acc_before))
        logging.info('Accuracy after: {}'.format(acc_after))
        logging.info('Number of iterations: {}'.format(total_iterations))
        logging.info('Fine-tuning in {}th iteration takes {:.2f}s'.format(t, time.time()-start))
        next_phase = None

    t += 1




from __future__ import division
from __future__ import print_function
import logging
import os
import time
import json
import sys
from pruning.objective_functions import alexnet_objective_function
from pruning.bayesian_optimization import bayesian_optimization
from pruning.utils import read_fp_log, find_next_phase

if len(sys.argv) == 1:
    resume_training = False
elif sys.argv[1] == 'resume':
    resume_training = True
else:
    raise ValueError('Command line argument error')

# hyper parameters
num_threads = 4
batch_size = 32
original_latency = 238
latency_constraint = 80
fine_pruning_iterations = 5
# for bayesian optimization
init_points = 30
bo_iters = 30
kappa = 10
cooling_function = 'linear'
min_acc = 0.55
max_iter = 100000
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1'

# some path variables
original_prototxt = 'models/bvlc_reference_caffenet/train_val.prototxt'
original_caffemodel = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
solver_file = 'models/bvlc_reference_caffenet/finetune_solver.prototxt'
output_folder = 'results/fp_{}_{}_{}'.format(fine_pruning_iterations, bo_iters, cooling_function)
# output_folder = 'results/bo/pts_{}_iter_{}_kappa_{}_to_{}'.format(init_points, bo_iters, kappa, 0.0625)
best_sampled_caffemodel = os.path.join(output_folder, 'best_sampled.caffemodel')
last_finetuned_caffemodel = os.path.join(output_folder, '0th_finetuned.caffemodel')
log_file = os.path.join(output_folder, 'fine_pruning.log')


def relaxed_constraint(iteration, cooling_func):
    if cooling_func == 'linear':
        return original_latency + (iteration+1)/fine_pruning_iterations * (latency_constraint - original_latency)


if resume_training:
    logging.basicConfig(filename=log_file, filemode='a+', level=logging.INFO)
    t, next_phase = find_next_phase(log_file)
    if next_phase != 'bayesian optimization':
        last_relaxed_constraint = relaxed_constraint(t-1, cooling_function)
    else:
        last_relaxed_constraint = relaxed_constraint(t, cooling_function)
elif os.path.exists(output_folder):
    raise IOError('{} already exist.'.format(output_folder))
else:
    os.mkdir(output_folder)
    logging.basicConfig(filename=log_file, filemode='a+', level=logging.INFO)
    t = 0
    next_phase = None
    last_relaxed_constraint = original_latency


while t < fine_pruning_iterations:
    if t == 0:
        input_caffemodel = original_caffemodel
    else:
        input_caffemodel = last_finetuned_caffemodel
    # compute relaxed constraints
    current_relaxed_constraint = relaxed_constraint(t, cooling_function)

    if next_phase is None or next_phase == 'bayesian optimization':
        logging.info('The relaxed constraint in {}th iteration is {}'.format(t, current_relaxed_constraint))
        logging.info('Start {}th fine-pruning iteration'.format(t))
        # first do bayesian optimization given latency tradeoff factor
        start = time.time()
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
        results = read_fp_log(log_file=log_file, bo_num=t)
        max_acc = 0
        max_res = None
        # TODO: what if there is no point sampled below the relaxed_constraint? increase the sampling point or ...?
        for res in results:
            if res.latency <= current_relaxed_constraint and res.accuracy > max_acc:
                max_res = res
                max_acc = res.accuracy
        logging.info('The best point chosen satisfying the constraint:')
        logging.info(max_res)

        # prune best point in sampled results
        start = time.time()
        pruning_dict_file = 'results/pruning_dict.txt'
        with open(pruning_dict_file, 'w') as fo:
            json.dump(max_res.pruning_dict, fo)
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
        # TODO: should min_acc, max_iter be a function of time?
        start = time.time()
        last_finetuned_caffemodel = os.path.join(output_folder, '{}th_finetuned.caffemodel'.format(t))
        finetuning_logfile = last_finetuned_caffemodel.replace('caffemodel', 'log')
        command = ['python', 'pruning/fine_tune.py', best_sampled_caffemodel, solver_file,
                   last_finetuned_caffemodel, str(min_acc), str(max_iter), finetuning_logfile]
        os.system(' '.join(command))
        logging.debug(' '.join(command))
        if not os.path.exists(last_finetuned_caffemodel):
            logging.error('Cannot find the finetuned caffemodel')
        logging.info('Fine-tuning in {}th iteration takes {:.2f}s'.format(t, time.time()-start))
        next_phase = None

    t += 1




from __future__ import division
from __future__ import print_function
import logging
import os
import time
import json
from pruning.objective_functions import alexnet_objective_function
from pruning.bayesian_optimization import bayesian_optimization
from pruning.utils import read_fp_log


# hyper parameters
num_threads = 4
batch_size = 32
original_latency = 238
latency_constraint = 80
fine_pruning_iterations = 5
bo_iters = 10
accuracy_lower_bound = 0.55
cooling_function = 'linear'
min_acc = 0.55
max_iter = 100
# allow 4 percent drop in accuracy to trade off for 140 ms speedup
# TODO: latency tradeoff factor should also be a function of time
latency_tradeoff = 4/140

# some path variables
original_prototxt = 'models/bvlc_reference_caffenet/train_val.prototxt'
original_caffemodel = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
solver_file = 'models/bvlc_reference_caffenet/finetune_solver.prototxt'
output_folder = 'results/fp_{}_{}_{}'.format(fine_pruning_iterations, bo_iters, cooling_function)
best_sampled_caffemodel = os.path.join(output_folder, 'best_sampled.caffemodel')
last_finetuned_caffemodel = os.path.join(output_folder, '0th_finetuned.caffemodel')

if os.path.exists(output_folder):
    raise IOError('{} already exist.'.format(output_folder))
else:
    os.mkdir(output_folder)

log_file = os.path.join(output_folder, 'fine_pruning.log')
logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)

# configure objective function
objective_function = alexnet_objective_function
objective_function.latency_tradeoff = latency_tradeoff
objective_function.original_latency = original_latency

for t in range(fine_pruning_iterations):
    logging.info('Start {}th fine-pruning iteration'.format(t))
    if t == 0:
        input_caffemodel = original_caffemodel
    else:
        input_caffemodel = last_finetuned_caffemodel

    # first do bayesian optimization given latency tradeoff factor
    start = time.time()
    objective_function.input_caffemodel = input_caffemodel
    bayesian_optimization(n_iter=bo_iters, tradeoff_factors=(latency_tradeoff,), output_folder=output_folder,
                          logged=False, objective_function=objective_function)
    logging.info('Bayesian optimization takes {:.2f}s'.format(time.time()-start))

    # compute relaxed constraints
    if cooling_function == 'linear':
        relaxed_constraint = original_latency + (t+1)/fine_pruning_iterations * (latency_constraint - original_latency)
    logging.info('The relaxed constraint in {}th iteration is {}'.format(t, relaxed_constraint))

    # find the best point satisfying the relaxed constraints
    results = read_fp_log(log_file=log_file, bo_num=t)
    max_acc = 0
    max_res = None
    # TODO: what if there is no point sampled below the relaxed_constraint? increase the sampling point or ...?
    for res in results:
        if res.latency <= relaxed_constraint and res.accuracy > max_acc:
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
    logging.info('Pruning takes {:.2f}s'.format(time.time()-start))

    # avoid affecting latency measurement, run fine-tuning and pruning from command line
    # fine-tune the pruned caffemodel until acc > min_acc or iteration > max_iter
    # TODO: should min_acc, max_iter be a function of time?
    start = time.time()
    last_finetuned_caffemodel = os.path.join(output_folder, '{}th_finetuned.caffemodel'.format(t))
    finetuning_logfile = os.path.join(output_folder, last_finetuned_caffemodel.replace('caffemodel', 'log'))
    command = ['python', 'pruning/fine_tune.py', best_sampled_caffemodel, solver_file,
               last_finetuned_caffemodel, str(min_acc), str(max_iter), finetuning_logfile]
    os.system(' '.join(command))
    logging.info('Fine-tuning takes {:.2f}s'.format(time.time()-start))

print('Final result', max_acc)



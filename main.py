from __future__ import division
from __future__ import print_function

import logging
import os
import time
from pruning.objective_functions import alexnet_objective_function
from pruning.bayesian_optimization import bayesian_optimization
from pruning.fine_tuning import fine_tuning
from pruning.utils import read_log
from pruning.prune import prune


# hyper parameters
num_threads = 4
batch_size = 32
original_latency = 238
latency_constraint = 80
fine_pruning_iterations = 5
bo_iters = 200
accuracy_lower_bound = 0.55
cooling_function = 'linear'
# allow 4 percent drop in accuracy to trade off for 140 ms speedup
# TODO: latency tradeoff factor should also be a function of time
latency_tradeoff = 4/140

# some path variables
original_prototxt = 'models/bvlc_reference_caffenet/train_val.prototxt'
original_caffemodel = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
best_sampled_caffemodel = 'results/best_sampled.caffemodel'
output_folder = 'results/fp_{}_{}_{}'.format(fine_pruning_iterations, bo_iters, cooling_function)
log_file = os.path.join(output_folder, 'fine_pruning.log')
logging.basicConfig(log_file=log_file, filemode='w', level=logging.INFO)

# configure objective function
objective_function = alexnet_objective_function
objective_function.latency_tradeoff = latency_tradeoff
objective_function.original_latency = original_latency
last_finetuned_caffemodel = None

for t in range(fine_pruning_iterations):
    if t == 0:
        input_caffemodel = original_caffemodel
    else:
        input_caffemodel = last_finetuned_caffemodel

    # first do bayesian optimization given latency tradeoff factor
    start = time.time()
    objective_function.input_caffemodel = input_caffemodel
    bayes_log = bayesian_optimization(n_iter=bo_iters, tradeoff_factors=(latency_tradeoff,),
                                      output_folder=output_folder, bo_num=t, objective_function=objective_function)
    print('Bayesian optimization takes {:.2f}s'.format(time.time()-start))

    # compute relaxed constraints
    if cooling_function == 'linear':
        relaxed_constraint = original_latency + (t+1)/fine_pruning_iterations * (latency_constraint - original_latency)

    # find the best point satisfying the constraints
    results = read_log(bayes_log)
    max_acc = 0
    max_res = None
    for res in results:
        if res.latency <= relaxed_constraint and res.accuracy > max_acc:
            max_res = res
            max_acc = res.accuracy

    # prune and fine-tune the best point for until accuracy recovers to a lower bound of accuracy
    # TODO: study fine-tune to k iterations or set a hard constraint or make it a function of t or till converge
    start = time.time()
    prune(input_caffemodel, original_prototxt, best_sampled_caffemodel, max_res.pruning_dict)
    print('Pruning takes {:.2f}s'.format(time.time()-start))
    start = time.time()
    fine_tuning(best_sampled_caffemodel, last_finetuned_caffemodel)
    print('Fine-tuning takes {:.2f}s'.format(time.time()-start))

print('Final result', max_acc)



import logging
import time
from calculate_objective import alexnet_target_function
import itertools


def binary_search(n_iter):
    # binary search with different beta values
    alexnet_range = {'conv1': (0, 1), 'conv2': (0, 1), 'conv3': (0, 1), 'conv4': (0, 1),
                     'conv5': (0, 1), 'fc6': (0, 1), 'fc7': (0, 1), 'fc8': (0, 1)}
    logging.basicConfig(filename='results/binary_{}.log'.format(n_iter), filemode='w', level=logging.INFO)

    target_function = alexnet_target_function
    target_function.latency_tradeoff = 50
    range_dict = alexnet_range
    candidate_counter = 0
    best_objective = 0
    best_pruning_dict = None
    candidate_pruning_list = generate_candidates(range_dict, best_pruning_dict)
    for i in range(n_iter):
        start = time.time()
        if candidate_counter == 256:
            candidate_counter = 0
            candidate_pruning_list = generate_candidates(range_dict, best_pruning_dict)

        # search the binary pruning list
        pruning_dict = candidate_pruning_list[candidate_counter]
        objective = target_function(**pruning_dict)
        if objective > best_objective:
            best_pruning_dict = pruning_dict
        candidate_counter += 1
        print('Iteration {} takes {:.2f}s'.format(i, time.time()-start))


def generate_candidates(range_dict, best_pruning_dict):
    # range_dict stores the range of each layer
    # update the range_dict according to best pruning result in the last round
    if best_pruning_dict is not None:
        for layer in best_pruning_dict:
            r = range_dict[layer]
            if best_pruning_dict[layer] == 3*r[0]/4 + r[1]/4:
                range_dict[layer][1] = (r[0] + r[1]) / 2
            else:
                range_dict[layer][0] = (r[0] + r[1]) / 2

    # generate next 256 candidate pruning_dict for searching
    candidate_pruning_list = []
    layer_candidates = []
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    for layer in layers:
        r = range_dict[layer]
        layer_candidates.append([3*r[0]/4 + r[1]/4, r[0]/4 + 3*r[1]/4])
    all_candidates = list(itertools.product(*layer_candidates))
    for candidate in all_candidates:
        pruning_dict = {layers[i]: p for i, p in enumerate(candidate)}
        candidate_pruning_list.append(pruning_dict)
    return candidate_pruning_list


if __name__ == '__main__':
    binary_search(n_iter=749)

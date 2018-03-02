import random
import logging
import time
from objective_functions import objective_function


def random_search(n_iter):
    alexnet_range = {'conv1': (0, 1), 'conv2': (0, 1), 'conv3': (0, 1), 'conv4': (0, 1),
                     'conv5': (0, 1), 'fc6': (0, 1), 'fc7': (0, 1), 'fc8': (0, 1)}
    logging.basicConfig(filename='results/pre_random_{}.log'.format(n_iter), filemode='w', level=logging.INFO)
    objective_function.latency_tradeoff = 50
    for i in range(n_iter):
        start = time.time()
        # random pruning percentage in range [0, 1) for each layer in alexnet
        pruning_dict = {layer: random.random() for layer in alexnet_range}
        objective_function(**pruning_dict)
        print('Iteration {} takes: {:.2f}s'.format(i, time.time()-start))


if __name__ == '__main__':
    random_search(n_iter=1000)

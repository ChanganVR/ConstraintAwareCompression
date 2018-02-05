import logging
import os
from bayes_opt import BayesianOptimization


def bayesian_optimization(n_iter=1000, tradeoff_factors=(1,), objective_function=None, init_points=10, kappa=5):
    local_n_iter = int(n_iter / len(tradeoff_factors))
    for tradeoff_factor in tradeoff_factors:
        logging.info('Bayesian optimization tradeoff factor: {:.4f}'.format(tradeoff_factor))
        bo = BayesianOptimization(objective_function,
                                  {'conv1': (0, 1), 'conv2': (0, 1), 'conv3': (0, 1), 'conv4': (0, 1),
                                   'conv5': (0, 1), 'fc6': (0, 1), 'fc7': (0, 1), 'fc8': (0, 1)})

        bo.maximize(init_points=init_points, n_iter=local_n_iter, kappa=kappa)


if __name__ == '__main__':
    iterations = 1000
    tradeoff = [1, 2, 4, 16, 32]

    # if len(tradeoff) > 1:
    #     filename = 'results/pre_mbo_{}_{}_{}.log'.format(init_points, iterations, kappa)
    # else:
    #     filename = 'results/bo_{}_{}_{}.log'.format(init_points, iterations, kappa)
    # logging.basicConfig(filename='', filemode='w', level=logging.INFO)
    # bayesian_optimization(n_iter=iterations, tradeoff_factors=tradeoff)

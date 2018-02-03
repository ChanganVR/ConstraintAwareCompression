import logging
import os
from bayes_opt import BayesianOptimization


def bayesian_optimization(n_iter=1000, tradeoff_factors=(1,), output_folder='results', bo_num=None, objective_function=None):
    # set some default parameters for bayesian optimization
    init_points = 10
    kappa_decay = True
    kappa_range = (2, 10)

    if len(tradeoff_factors) > 1:
        filename = 'pre_mbo_{}_{}_{}_{}.log'.format(init_points, n_iter, kappa_range[1], kappa_range[0])
        filename = os.path.join(output_folder, filename)
    else:
        if bo_num is None:
            filename = 'bo_{}_{}_{}_{}.log'.format(init_points, n_iter, kappa_range[1], kappa_range[0])
        else:
            filename = '{}_bo_{}_{}_{}_{}.log'.format(bo_num, init_points, n_iter, kappa_range[1], kappa_range[0])
        filename = os.path.join(output_folder, filename)

    logging.basicConfig(filename=filename, filemode='w', level=logging.INFO)

    local_n_iter = int(n_iter / len(tradeoff_factors))
    for tradeoff_factor in tradeoff_factors:
        print('Switch to tradeoff factor {}...'.format(tradeoff_factor))
        logging.info('Switch to tradeoff factor {}...'.format(tradeoff_factor))
        bo = BayesianOptimization(objective_function,
                                  {'conv1': (0, 1), 'conv2': (0, 1), 'conv3': (0, 1), 'conv4': (0, 1),
                                   'conv5': (0, 1), 'fc6': (0, 1), 'fc7': (0, 1), 'fc8': (0, 1)})

        if not kappa_decay:
            bo.maximize(init_points=init_points, n_iter=local_n_iter, kappa=5)
        else:
            bo.maximize(init_points=init_points, n_iter=0, kappa=5)
            for i in range(1, local_n_iter+1):
                current_kappa = kappa_range[1] + (kappa_range[0] - kappa_range[1]) / local_n_iter * i
                bo.maximize(init_points=0, n_iter=1, kappa=current_kappa)

    return filename


if __name__ == '__main__':
    bayesian_optimization(n_iter=1000, tradeoff_factors=[1, 2, 4, 16, 32])

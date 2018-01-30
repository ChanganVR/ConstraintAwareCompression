import logging
from bayes_opt import BayesianOptimization
from objective_functions import alexnet_objective_function


def bayesian_optimization(init_points, n_iter, tradeoff_factors, kappa_decay=True, kappa_range=(2, 10)):
    if len(tradeoff_factors) > 1:
        filename = 'results/mbo_{}_{}_{}_{}.log'.format(init_points, n_iter, kappa_range[1], kappa_range[0])
    else:
        filename = 'results/bo_{}_{}_{}_{}.log'.format(init_points, n_iter, kappa_range[1], kappa_range[0])
    logging.basicConfig(filename=filename, filemode='w', level=logging.INFO)

    local_n_iter = int(n_iter / len(tradeoff_factors))
    for tradeoff_factor in tradeoff_factors:
        print('Switch to tradeoff factor {}...'.format(traoff_factor))
        objective_function = alexnet_objective_function
        objective_function.latency_tradeoff = tradeoff_factor
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

        # Finally, we take a look at the final results.
        # logging.info(bo.res['max'])
        # logging.info(bo.res['all'])


if __name__ == '__main__':
    bayesian_optimization(init_points=10, n_iter=1000, tradeoff_factors=[1, 2, 4, 16, 32])

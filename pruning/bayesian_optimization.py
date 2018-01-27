import logging
from bayes_opt import BayesianOptimization
from calculate_objective import alexnet_target_function


def optimize(kappa_decay):
    # Lets find the maximum of a simple quadratic function of two variables
    # We create the bayes_opt object and pass the function to be maximized
    # together with the parameters names and their bounds.
    bo = BayesianOptimization(alexnet_target_function,
                              {'conv1': (0, 1), 'conv2': (0, 1), 'conv3': (0, 1), 'conv4': (0, 1),
                               'conv5': (0, 1), 'fc6': (0, 1), 'fc7': (0, 1), 'fc8': (0, 1)})

    # One of the things we can do with this object is pass points
    # which we want the algorithm to probe. A dictionary with the
    # parameters names and a list of values to include in the search
    # must be given.
    # bo.explore({'x': [-1, 3], 'y': [-2, 2]})

    # Additionally, if we have any prior knowledge of the behaviour of
    # the target function (even if not totally accurate) we can also
    # tell that to the optimizer.
    # Here we pass a dictionary with 'target' and parameter names as keys and a
    # list of corresponding values

    # bo.initialize(
    #     {
    #         'target': [-1, -1],
    #         'x': [1, 1],
    #         'y': [0, 2]
    #     }
    # )

    # Once we are satisfied with the initialization conditions
    # we let the algorithm do its magic by calling the maximize() method.
    init_points = 10
    n_iter = 1200
    kappa_upper = 10
    kappa_lower = 1
    kappa_decay = True
    if not kappa_decay:
        logging.basicConfig(filename='results/pruning_{}_{}_{}.log'.format(init_points, n_iter, kappa_upper),
                            filemode='w', level=logging.INFO)
        bo.maximize(init_points=10, n_iter=80, kappa=5)
    else:
        logging.basicConfig(filename='results/pruning_{}_{}_{}_{}.log'.format(init_points, n_iter, kappa_upper, kappa_lower),
                            filemode='w', level=logging.INFO)
        bo.maximize(init_points=10, n_iter=0, kappa=5)
        for i in range(n_iter):
            current_kappa = kappa_upper + (kappa_lower - kappa_upper) / n_iter * (i+1)
            bo.maximize(init_points=0, n_iter=1, kappa=current_kappa)

    # The output values can be accessed with self.res
    # logging.info(bo.res['max'])

    # If we are not satisfied with the current results we can pickup from
    # where we left, maybe pass some more exploration points to the algorithm
    # change any parameters we may choose, and the let it run again.
    # bo.explore({'x': [0.6], 'y': [-0.23]})

    # Making changes to the gaussian process can impact the algorithm
    # dramatically.
    gp_params = {'kernel': None,
                 'alpha': 1e-5}

    # Run it again with different acquisition function
    # bo.maximize(n_iter=5, acq='ei', **gp_params)

    # Finally, we take a look at the final results.
    logging.info(bo.res['max'])
    logging.info(bo.res['all'])
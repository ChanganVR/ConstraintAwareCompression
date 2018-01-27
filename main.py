from pruning.bayesian_optimization import optimize
import logging

logging.basicConfig(filename='results/pruning.log', filemode='w', level=logging.INFO)

optimize()

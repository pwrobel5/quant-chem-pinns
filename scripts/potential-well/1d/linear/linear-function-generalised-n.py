import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.continuousn as continuousn
import quantchem.pinns.problems.linearwell as linearwell


N_MAX = 5

L = 2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ndense', '--layers', default=defaults.DEFAULT_LAYERS)
    parser.add_argument('-nnodes', '--nodes', default=defaults.DEFAULT_NODES)
    parser.add_argument('-ntrain', '--num-train', default=defaults.DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=defaults.DEFAULT_NUM_TEST)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train) ** 2
    num_test = int(args.num_test) ** 2

    problem = linearwell.LinearWellVariableN(1, N_MAX, L, 0)

    function_net = continuousn.FunctionContinuousN(problem.exact_solution, problem.domain, layers, nodes, num_train, num_test)
    function_net.train_net()

    storage.save_loss_plot('linear-function-generalised-n')
    
    for i in range(1, 8):
        storage.save_results_variable_n_function(i, function_net, problem.exact_solution, -L / 2, L / 2, args.num_test, layers, nodes, num_train, num_test, 'linear-function-generalised-n')

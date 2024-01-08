import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet
import quantchem.pinns.problems.linearwell as linearwell


N_MAX = 5

L = 2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ndense', '--layers', default=defaults.DEFAULT_LAYERS)
    parser.add_argument('-nnodes', '--nodes', default=defaults.DEFAULT_NODES)
    parser.add_argument('-ntrain', '--num-train', default=defaults.DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=defaults.DEFAULT_NUM_TEST)
    parser.add_argument('-weights', '--weights', default=defaults.DEFAULT_LOSS_WEIGHTS)
    parser.add_argument('-random', '--random-collocation-points', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train) ** 2
    num_test = int(args.num_test) ** 2
    num_boundary = N_MAX * 2
    weights = int(args.weights)
    is_random = True if args.random_collocation_points else False

    problem = linearwell.LinearWellVariableN(1, N_MAX, L, weights, is_random)

    pde_net = pdenet.PDEApproach(problem.exact_solution, problem.domain, problem.pde, problem.boundary_conditions, problem.loss_weights, layers, nodes, num_train, num_boundary, num_test, input_dimension=2)
    pde_net.train_net()

    storage.save_loss_plot('linear-generalised-n')

    for i in range(1, 8):
        storage.save_results_variable_n_pde(i, pde_net, problem.exact_solution, -L / 2, L / 2, args.num_test,
                                            layers, nodes, num_train, num_test, is_random, weights,
                                            'linear-generalised-n')

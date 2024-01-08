import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet
import quantchem.pinns.problems.linearwell as linearwell


DEFAULT_N = 1

L = 2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--quantum-number', default=DEFAULT_N)
    parser.add_argument('-ndense', '--layers', default=defaults.DEFAULT_LAYERS)
    parser.add_argument('-nnodes', '--nodes', default=defaults.DEFAULT_NODES)
    parser.add_argument('-ntrain', '--num-train', default=defaults.DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=defaults.DEFAULT_NUM_TEST)
    parser.add_argument('-weights', '--weights', default=defaults.DEFAULT_LOSS_WEIGHTS)
    parser.add_argument('-random', '--random-collocation-points', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    n = int(args.quantum_number)
    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)
    num_boundary = 2
    weights = int(args.weights)
    is_random = True if args.random_collocation_points else False

    problem = linearwell.LinearWell(n, L, weights, is_random)

    pde_net = pdenet.PDEApproach(problem.exact_solution, problem.domain, problem.pde, problem.boundary_conditions, problem.loss_weights, layers, nodes, num_train, num_boundary, num_test)
    pde_net.train_net()

    storage.save_loss_plot('linear-pde')

    function_name = '$\psi_{}$(x)'.format(n)
    plot_file_name = '{}-{}-{}-{}-{}-{}'.format(n, layers, nodes, num_train, is_random, weights)
    storage.save_prediction_plot(function_name, plot_file_name)
    
    test_metric = pde_net.get_test_metric()
    csv_row = [n, layers, nodes, num_train, num_test, is_random, weights, test_metric]
    storage.save_to_csv('linear-pde', csv_row)

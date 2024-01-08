import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.fixedn as fixedn
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

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    n = int(args.quantum_number)
    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)

    problem = linearwell.LinearWell(n, L, 0)
        
    function_net = fixedn.FunctionFixedN(problem.exact_solution, problem.domain, layers, nodes, num_train, num_test)
    function_net.train_net()
    
    storage.save_loss_plot('linear-function')

    function_name = '$\psi_{}$(x)'.format(n)
    plot_file_name = '{}-{}-{}-{}'.format(n, layers, nodes, num_train)
    storage.save_prediction_plot(function_name, plot_file_name)

    test_metric = function_net.get_test_metric()
    csv_row = [n, layers, nodes, num_train, num_test, test_metric]
    storage.save_to_csv('linear-function', csv_row)

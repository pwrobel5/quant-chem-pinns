import deepxde as dde
import numpy as np
import scipy.special as sp
import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.fixedn as fixedn


DEFAULT_N = 1
DEFAULT_L = 0

Z = 1
R_MAX = 50


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-quantum-number', default=DEFAULT_N)
    parser.add_argument('-l', '--l-quantum-number', default=DEFAULT_L)
    parser.add_argument('-ndense', '--layers', default=defaults.DEFAULT_LAYERS)
    parser.add_argument('-nnodes', '--nodes', default=defaults.DEFAULT_NODES)
    parser.add_argument('-ntrain', '--num-train', default=defaults.DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=defaults.DEFAULT_NUM_TEST)

    return parser.parse_args()

def radial_part(r):
    normalization_constant = np.sqrt(
        np.power(2.0 / n, 3.0) * (np.math.factorial(n - l - 1) / (2 * n * np.math.factorial(n + l)))
    )
    
    exponent = np.exp(- (Z * r) / n)
    power = np.power(2 * Z * r / n, l)
    laguerre_polynomial = sp.genlaguerre(n - l - 1, 2 * l + 1)
    laguerre_value = laguerre_polynomial(2 * Z * r / n)
    
    return normalization_constant * exponent * power * laguerre_value


if __name__ == '__main__':
    args = parse_arguments()

    n = int(args.n_quantum_number)
    l = int(args.l_quantum_number)
    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)

    domain = dde.geometry.Interval(0, R_MAX)

    function_net = fixedn.FunctionFixedN(radial_part, domain, layers, nodes, num_train, num_test)
    function_net.train_net()

    storage.save_loss_plot('function')

    function_name = '$R_{{{},{}}}$(x)'.format(n, l)
    plot_file_name = '{}-{}-{}-{}-{}'.format(n, l, layers, nodes, num_train)
    storage.save_prediction_plot(function_name, plot_file_name)

    test_metric = function_net.get_test_metric()
    csv_row = [n, l, layers, nodes, num_train, num_test, test_metric]
    storage.save_to_csv('function', csv_row)

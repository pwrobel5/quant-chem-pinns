import deepxde as dde
import numpy as np
import scipy.special as sp
import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.fixedn as fixedn


DEFAULT_N = 1

R = 2
q = 1
epsilon = 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--quantum-number', default=DEFAULT_N)
    parser.add_argument('-ndense', '--layers', default=defaults.DEFAULT_LAYERS)
    parser.add_argument('-nnodes', '--nodes', default=defaults.DEFAULT_NODES)
    parser.add_argument('-ntrain', '--num-train', default=defaults.DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=defaults.DEFAULT_NUM_TEST)

    return parser.parse_args()

def psi(x):
    ai_zeros, _, _, _ = sp.ai_zeros(n)
    alpha_n = ai_zeros[-1]
    E = -alpha_n * np.power((q ** 2) * (epsilon ** 2) / 2, (1 / 3))
    
    cube_root = np.power(2 * q * epsilon, (1 / 3))
    
    ai_value, _, _, _ = sp.airy(cube_root * (x - (E / (q * epsilon))))
    _, aip_value, _, _ = sp.airy(-cube_root * (E / (q * epsilon)))
    
    normalization_constant = np.sqrt(cube_root / (aip_value ** 2))
    
    return normalization_constant * ai_value


if __name__ == '__main__':
    args = parse_arguments()

    n = int(args.quantum_number)
    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)

    domain = dde.geometry.Interval(0, R)

    function_net = fixedn.FunctionFixedN(psi, domain, layers, nodes, num_train, num_test)
    function_net.train_net()

    storage.save_loss_plot('triangular-function')

    function_name = '$\psi_{}$(x)'.format(n)
    plot_file_name = '{}-{}-{}-{}'.format(n, layers, nodes, num_train)
    storage.save_prediction_plot(function_name, plot_file_name)

    test_metric = function_net.get_test_metric()
    csv_row = [n, layers, nodes, num_train, num_test, test_metric]
    storage.save_to_csv('triangular-function', csv_row)

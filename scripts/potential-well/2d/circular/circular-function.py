import deepxde as dde
import numpy as np
import scipy.special as sp
import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.fixedn as fixedn

DEFAULT_N = 1
DEFAULT_L = 0

R = 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-quantum-number', default=DEFAULT_N)
    parser.add_argument('-l', '--l-quantum-number', default=DEFAULT_L)
    parser.add_argument('-ndense', '--layers', default=defaults.DEFAULT_LAYERS)
    parser.add_argument('-nnodes', '--nodes', default=defaults.DEFAULT_NODES)
    parser.add_argument('-ntrain', '--num-train', default=defaults.DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=defaults.DEFAULT_NUM_TEST)

    return parser.parse_args()

def angular(theta, l):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(theta * l * 1j)

def radial(r, n, l):
    alpha = sp.jn_zeros(l, n)[-1]
    normalization_constant = np.sqrt(2) / (R * np.abs(sp.jv(l + 1, alpha)))
    bessel = sp.jv(l, alpha * (r / R))

    return normalization_constant * bessel

def psi(x, n, l):
    r, theta = x[:, 0:1], x[:, 1:2]
    
    radial_part = radial(r, n, l)
    angular_part = angular(theta, l)

    return radial_part * angular_part


if __name__ == '__main__':
    args = parse_arguments()

    n = int(args.n_quantum_number)
    l = int(args.l_quantum_number)
    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)

    domain = dde.geometry.Interval(0, R)

    function_net = fixedn.FunctionFixedN(lambda x: radial(x, n, l), domain, layers, nodes, num_train, num_test)
    function_net.train_net()

    storage.save_loss_plot('circular-function')

    function_name = '$\psi_{{{},{}}}$(x)'.format(n, l)
    plot_file_name = '{}-{}-{}-{}-{}'.format(n, l, layers, nodes, num_train)
    storage.save_prediction_plot(function_name, plot_file_name)

    test_metric = function_net.get_test_metric()
    csv_row = [n, l, layers, nodes, num_train, num_test, test_metric]
    storage.save_to_csv('circular-function', csv_row)

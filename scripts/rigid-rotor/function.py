import deepxde as dde
import numpy as np
import scipy.special as sp
import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.fixedn as fixedn


DEFAULT_L = 1
DEFAULT_M = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--l-quantum-number', default=DEFAULT_L)
    parser.add_argument('-m', '--m-quantum-number', default=DEFAULT_M)
    parser.add_argument('-ndense', '--layers', default=defaults.DEFAULT_LAYERS)
    parser.add_argument('-nnodes', '--nodes', default=defaults.DEFAULT_NODES)
    parser.add_argument('-ntrain', '--num-train', default=defaults.DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=defaults.DEFAULT_NUM_TEST)

    return parser.parse_args()

def z_component(phi):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(phi * m * 1j)

def theta_part(theta):
    # 2*pi is already at the z_component
    normalization_constant = ((-1) ** m) * np.sqrt(((2 * l + 1) / 2) * (np.math.factorial(l - m) / np.math.factorial(l + m)))
    legendre_values = []
    for i in theta:
        legendre_polynomials, _ = sp.lpmn(m, l, np.cos(i[0]))
        legendre = legendre_polynomials[m, l]
        legendre_values.append([legendre])
    legendre_values = np.array(legendre_values)
    
    return normalization_constant * legendre_values

def spherical_harmonic(x):
    theta, phi = x[:, 0:1], x[:, 1:2]
    
    legendre_component = theta_part(theta)
    phi_component = z_component(phi)
    
    return legendre_component * phi_component


if __name__ == '__main__':
    args = parse_arguments()

    l = int(args.l_quantum_number)
    m = int(args.m_quantum_number)
    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)

    domain = dde.geometry.Interval(0, np.pi)

    function_net = fixedn.FunctionFixedN(theta_part, domain, layers, nodes, num_train, num_test)
    function_net.train_net()

    storage.save_loss_plot('function')

    function_name = '$\Theta_{{{},{}}}$(x)'.format(l, m)
    plot_file_name = '{}-{}-{}-{}-{}'.format(l, m, layers, nodes, num_train)
    storage.save_prediction_plot(function_name, plot_file_name)

    test_metric = function_net.get_test_metric()
    csv_row = [l, m, layers, nodes, num_train, num_test, test_metric]
    storage.save_to_csv('function', csv_row)

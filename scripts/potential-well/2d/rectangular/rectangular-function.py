import deepxde as dde
import numpy as np
import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.fixedn as fixedn
import quantchem.pinns.utils as utils


DEFAULT_NX = 1
DEFAULT_NY = 1

LX = 2
LY = 2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nx', '--nx-quantum-number', default=DEFAULT_NX)
    parser.add_argument('-ny', '--ny-quantum-number', default=DEFAULT_NY)
    parser.add_argument('-ndense', '--num-dense-layers', default=defaults.DEFAULT_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=defaults.DEFAULT_NODES)
    parser.add_argument('-ntrain', '--num-train', default=defaults.DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=defaults.DEFAULT_NUM_TEST)

    return parser.parse_args()

def psi(x, n, L):
    k = (n * np.pi) / L
    normalization_constant = np.sqrt(2.0 / L)
    return normalization_constant * np.sin(k * (x + 0.5 * L))


if __name__ == '__main__':
    args = parse_arguments()

    nx = int(args.nx_quantum_number)
    ny = int(args.ny_quantum_number)
    layers = int(args.num_dense_layers)
    nodes = int(args.num_dense_nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)

    domain_x = dde.geometry.Interval(-LX / 2, LX / 2)
    domain_y = dde.geometry.Interval(-LY / 2, LY / 2)

    function_net_x = fixedn.FunctionFixedN(lambda x: psi(x, nx, LX), domain_x, layers, nodes, num_train, num_test)
    function_net_y = fixedn.FunctionFixedN(lambda y: psi(y, ny, LY), domain_y, layers, nodes, num_train, num_test)
    
    function_net_x.train_net()
    function_net_y.train_net()   

    true_values = utils.get_2d_values(
        lambda data: utils.value_2d_rectangle(data, lambda x: psi(x, nx, LX), lambda y: psi(y, ny, LY)),
        -LX / 2,
        LX / 2,
        -LY / 2,
        LY / 2
    )
    
    predicted_values = utils.get_2d_values(
        lambda x: utils.value_2d_model(x, function_net_x, function_net_y),
        -LX / 2,
        LX / 2,
        -LY / 2,
        LY / 2
    )

    extent=[-LX / 2, LX / 2, -LY / 2, LY / 2]
    output_name_body = '{}-{}-{}-{}-{}'.format(nx, ny, layers, nodes, num_train)

    storage.plot_2d_map(
        predicted_values, 
        '$\psi_{{{}, {}}}$(x, y)'.format(nx, ny), 
        [-LX / 2, LX / 2, -LY / 2, LY / 2], 
        'prediction-{}'.format(output_name_body)
    )
    storage.plot_2d_map(
        np.abs(true_values - predicted_values), 
        '|$\psi_{{{},{}, predicted}}$(x, y) - $\psi_{{{},{}, true}}$(x, y)|'.format(nx, ny, nx, ny), 
        [-LX / 2, LX / 2, -LY / 2, LY / 2], 
        'difference-{}'.format(output_name_body)
    )

    test_metric = dde.metrics.l2_relative_error(true_values, predicted_values)
    csv_row = [nx, ny, layers, nodes, num_train, num_test, test_metric]
    storage.save_to_csv('rectangular-function', csv_row)

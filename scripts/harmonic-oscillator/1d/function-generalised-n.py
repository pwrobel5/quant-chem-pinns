import deepxde as dde
import numpy as np
import scipy.special as sp
import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.utils as utils
import quantchem.pinns.approaches.function.continuousn as continuousn


N_MAX = 5

L = 5
m = 1
omega = 0.5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ndense', '--layers', default=defaults.DEFAULT_LAYERS)
    parser.add_argument('-nnodes', '--nodes', default=defaults.DEFAULT_NODES)
    parser.add_argument('-ntrain', '--num-train', default=defaults.DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=defaults.DEFAULT_NUM_TEST)

    return parser.parse_args()

def psi(x):
    n = x[:, 1:2]
    factorials = []
    for n_val in n:
        factorials.append(sp.gamma(n_val + 1.0))
    factorials = np.array(factorials).reshape(len(factorials), 1)
    
    constants = (1.0 / (np.sqrt(factorials * (2 ** n)))) * (((m * omega) / np.pi) ** 0.25)

    exponent = np.exp(-0.5 * m * omega * np.power(x[:, 0:1], 2))
    
    hermite_values = []
    for index, n_val in enumerate(n):
        coefficients = int(n_val) * [0] + [1]
        hermite = np.polynomial.hermite.Hermite(coefficients)
        hermite_value = hermite(x[index, 0] * np.sqrt(m * omega))
        hermite_values.append(hermite_value)
    hermite_values = np.array(hermite_values).reshape(len(hermite_values), 1)

    result = constants * exponent * hermite_values
    return result


if __name__ == '__main__':
    args = parse_arguments()

    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train) ** 2
    num_test = int(args.num_test) ** 2

    domain = dde.geometry.Rectangle([-L, 0], [L, N_MAX])

    function_net = continuousn.FunctionContinuousN(psi, domain, layers, nodes, num_train, num_test)
    function_net.train_net()
    
    storage.save_loss_plot('function-generalised-n')
    
    for i in range(0, 7):
        x, y_pred, y_true = utils.get_values_for_n(i, function_net, psi, -L, L, defaults.DEFAULT_NUM_TEST)
        function_name = '$\psi_{}$(x)'.format(i)
        plot_file_name = '{}-{}-{}-{}'.format(i, layers, nodes, num_train)
        storage.save_prediction_plot_from_points(function_name, plot_file_name,
                                                 x, x, y_true, y_true, y_pred,
                                                 train_label='Testing points')
        
        test_metric = dde.metrics.l2_relative_error(y_true, y_pred)
        csv_row = [i, layers, nodes, num_train, num_test, test_metric]
        storage.save_to_csv('function-generalised-n', csv_row)

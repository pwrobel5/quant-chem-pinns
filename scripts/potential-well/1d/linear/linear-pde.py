import deepxde as dde
import numpy as np
import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet


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

def pde(x, y):
    k = (n * np.pi) / L
    E = 0.5 * (k ** 2)
    dy_xx = dde.grad.hessian(y, x)
    return 0.5 * dy_xx + E * y

def psi(x):
    k = (n * np.pi) / L
    normalization_constant = np.sqrt(2.0 / L)
    return normalization_constant * np.sin(k * (x + 0.5 * L))

def get_extremal_collocation_points(n):
    points = []
    for k in range(n):
        x = (k * L) / n + L / (2 * n) - L / 2
        points.append([x])
        
    if n == 1:
        points.append([L / 4])
        points.append([-L / 4])
    
    return np.array(points)

def get_random_collocation_points(n):
    points = []
    for _ in range(n):
        x = np.random.uniform(low=-L/2, high=L/2)
        points.append([x])
    
    if n == 1:
        points.append([np.random.uniform(low=-L/2, high=L/2)])
        points.append([np.random.uniform(low=-L/2, high=L/2)])
    
    return np.array(points)

def x_boundary(_, on_boundary):
    return on_boundary


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

    domain = dde.geometry.Interval(-L / 2, L / 2)

    collocation_points = get_random_collocation_points(n) if is_random else get_extremal_collocation_points(n)
    collocation_values = psi(collocation_points)

    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)
    dirichlet_bc = dde.icbc.DirichletBC(domain, lambda x: 0, x_boundary)

    boundary_conditions = [ic, dirichlet_bc]
    loss_weights = [1, weights, weights]

    pde_net = pdenet.PDEApproach(psi, domain, pde, boundary_conditions, loss_weights, layers, nodes, num_train, num_boundary, num_test)
    pde_net.train_net()

    storage.save_loss_plot('linear-pde')

    function_name = '$\psi_{}$(x)'.format(n)
    plot_file_name = '{}-{}-{}-{}-{}-{}'.format(n, layers, nodes, num_train, is_random, weights)
    storage.save_prediction_plot(function_name, plot_file_name)
    
    test_metric = pde_net.get_test_metric()
    csv_row = [n, layers, nodes, num_train, num_test, is_random, weights, test_metric]
    storage.save_to_csv('linear-pde', csv_row)

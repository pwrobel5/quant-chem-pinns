import deepxde as dde
import numpy as np
import argparse

import quantchem.pinns.approaches.defaults as defaults
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.utils as utils
import quantchem.pinns.approaches.pde.pde as pdenet


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

def pde(x, y):
    n = x[:, 1:2]
    k = (n * np.pi) / L
    E = 0.5 * (k ** 2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return 0.5 * dy_xx + E * y

def psi(x):
    normalization_constant = np.sqrt(2.0 / L)
    n = x[:, 1:2]
    k = (n * np.pi) / L
    return normalization_constant * np.sin(k * (x[:, 0:1] + (L / 2)))

def get_extremal_collocation_points(n_max):
    points = []
    for n in range(1, n_max + 1):
        for k in range(n):
            x = (k * L) / n + L / (2 * n) - L / 2
            points.append((x, n))
            
        if n == 1:
            points.append((L / 4, n))
            points.append((-L / 4, n))
    
    return np.array(points)

def get_random_collocation_points(n_max):
    points = []
    for n in range(1, n_max + 1):
        for _ in range(n):
            x = np.random.uniform(low=-L/2, high=L/2)
            points.append((x, n))
        
        if n == 1:
            points.append((np.random.uniform(low=-L/2, high=L/2), n))
            points.append((np.random.uniform(low=-L/2, high=L/2), n))
    
    return np.array(points)

def x_boundary(x, on_boundary):
    if x[1].is_integer():
        return np.isclose(-L / 2, x[0]) or np.isclose(L / 2, x[0])
    return False


if __name__ == '__main__':
    args = parse_arguments()

    layers = int(args.layers)
    nodes = int(args.nodes)
    num_train = int(args.num_train) ** 2
    num_test = int(args.num_test) ** 2
    num_boundary = N_MAX * 2
    weights = int(args.weights)
    is_random = True if args.random_collocation_points else False

    domain = dde.geometry.Rectangle([-L / 2, 1], [L / 2, N_MAX])

    collocation_points = get_random_collocation_points(N_MAX) if is_random else get_extremal_collocation_points(N_MAX)
    collocation_values = psi(collocation_points)

    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)
    dirichlet_bc = dde.icbc.DirichletBC(domain, lambda x: 0, x_boundary)

    boundary_conditions = [ic, dirichlet_bc]
    loss_weights = [1, weights, weights]

    pde_net = pdenet.PDEApproach(psi, domain, pde, boundary_conditions, loss_weights, layers, nodes, num_train, num_boundary, num_test, input_dimension=2)
    pde_net.train_net()

    storage.save_loss_plot('linear-generalised-n')

    for i in range(1, 8):
        storage.save_results_variable_n_pde(i, pde_net, psi, -L / 2, L / 2, args.num_test,
                                            layers, nodes, num_train, num_test, is_random, weights,
                                            'linear-generalised-n')

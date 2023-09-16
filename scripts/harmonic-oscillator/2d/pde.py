import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 60
DEFAULT_NX = 0
DEFAULT_NY = 0
DEFAULT_NUM_TRAIN = 64
DEFAULT_NUM_TEST = 100
DEFAULT_WEIGHTS = 1

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY = 2

LX = 5
LY = 5
m = 1
omega_x = 0.5
omega_y = 0.5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nx', '--nx-quantum-number', default=DEFAULT_NX)
    parser.add_argument('-ny', '--ny-quantum-number', default=DEFAULT_NY)
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)
    parser.add_argument('-weights', '--weights', default=DEFAULT_WEIGHTS)

    return parser.parse_args()

def pde(x, y, n, omega):
    dy_xx = dde.grad.hessian(y, x)
    E = (n + 0.5) * omega
    U = 0.5 * m * (omega ** 2) * (x ** 2)
    return -dy_xx / 2 + (U - E) * y

def psi(x, n, omega):
    constants = (1.0 / (np.sqrt(np.math.factorial(n) * (2 ** n)))) * (((m * omega) / np.pi) ** 0.25)
    exponent = np.exp(-0.5 * m * omega * np.power(x, 2))
    hermite_coefficients = [0] * n + [1]
    hermite = np.polynomial.hermite.Hermite(hermite_coefficients)
    hermite_value = hermite(x * np.sqrt(m * omega))
    result = constants * exponent * hermite_value
    
    return result

def x_boundary_even(x, _):
    return np.isclose(x[0], 0)

def x_boundary_odd(x, L, _):
    return np.isclose(x[0], L / 2) or np.isclose(x[0], -L / 2)

def boundary_value(n, omega, _):
    if n % 2 == 1:
        return 0
    
    constants = (1.0 / (np.sqrt(np.math.factorial(n) * (2 ** n)))) * (((m * omega) / np.pi) ** 0.25)
    if n == 0:
        return constants
    
    hermite_coefficients = [0] * n + [1]
    hermite = np.polynomial.Hermite(hermite_coefficients)
    hermite_value = hermite(0)

    return constants * hermite_value

def boundary_derivative_value(n, omega, _):
    if n % 2 == 0:
        return 0
    
    constants = 2 * n * (1.0 / (np.sqrt(np.math.factorial(n) * (2 ** n)))) * (1.0 / (np.pi ** 0.25)) * ((m * omega) ** 0.75)
    hermite_coefficients = [0] * (n - 1) + [1]
    hermite = np.polynomial.Hermite(hermite_coefficients)
    hermite_value = hermite(0)

    return constants * hermite_value

def save_to_csv():
    row = [nx, ny, num_dense_layers, num_dense_nodes, num_train, num_test, weights, test_metric]
    
    csv_file = open('pde.csv', 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()

def train_1d_nn(n, L, omega):
    domain = dde.geometry.Interval(-L, L)

    if n % 2 == 0:
        dirichlet_bc = dde.icbc.DirichletBC(domain, lambda x: boundary_value(n, omega, x), x_boundary_even)
    else:
        dirichlet_bc = dde.icbc.DirichletBC(domain, lambda x: psi(x, n, omega), lambda x, on_boundary: x_boundary_odd(x, L, on_boundary))
    neumann_bc = dde.icbc.NeumannBC(domain, lambda x: boundary_derivative_value(n, omega, x), x_boundary_even)

    data = dde.data.PDE(
        domain, 
        lambda x, y: pde(x, y, n, omega), 
        [dirichlet_bc, neumann_bc], 
        num_domain=num_train, 
        num_boundary=NUM_BOUNDARY,
        solution=lambda x: psi(x, n, omega), 
        num_test=num_test
    )

    net = dde.nn.FNN(
        [1] + [num_dense_nodes] * num_dense_layers + [1], 
        ACTIVATION, 
        INITIALIZER
    )

    model = dde.Model(data, net)
    loss_weights = [1, weights, weights]
    model.compile(
        OPTIMIZER, 
        metrics=METRICS,
        loss_weights=loss_weights
    )

    model.train(iterations=ITERATIONS)

    return model

def psi_2d_rectangle(data, nx, ny):
    x_values = data[:, 0:1]
    y_values = data[:, 1:2]

    x_component = psi(x_values, nx, omega_x)
    y_component = psi(y_values, ny, omega_y)

    return x_component * y_component

def psi_2d_model(data, x_model, y_model):
    x_values = data[:, 0:1]
    y_values = data[:, 1:2]

    x_component = x_model.predict(x_values)
    y_component = y_model.predict(y_values)

    return x_component * y_component

def get_2d_values(input_function, grid_dist=0.02):
    x_ax = np.arange(-LX, LX, grid_dist)
    y_ax = np.arange(-LY, LY, grid_dist)
    grid_x, grid_y = np.meshgrid(x_ax, y_ax)
    
    function_list = []
    for a, b in zip(grid_x, grid_y):
        pair_list = []
        for x, y in zip(a, b):
            pair_list.append([x, y])
        pairs = np.array(pair_list)
        function_list.append(np.squeeze(input_function(pairs)))
    
    function_values = np.array(function_list)
    return function_values

def plot_2d_map(function_values, zlabel='abc', output_prefix='function'):
    plt.figure()
    im = plt.imshow(function_values, cmap=plt.cm.RdBu, extent=[-LX, LX, -LY, LY])
    plt.colorbar(im, label=zlabel)
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.savefig('{}-{}-{}-{}-{}-{}-results.png'.format(output_prefix, nx, ny, num_dense_layers, num_dense_nodes, num_train))


if __name__ == '__main__':
    args = parse_arguments()

    nx = int(args.nx_quantum_number)
    ny = int(args.ny_quantum_number)
    num_dense_layers = int(args.num_dense_layers)
    num_dense_nodes = int(args.num_dense_nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)
    weights = int(args.weights)

    model_x = train_1d_nn(nx, LX, omega_x)
    model_y = train_1d_nn(ny, LY, omega_y)

    true_values = get_2d_values(lambda x: psi_2d_rectangle(x, nx, ny))

    predicted_values = get_2d_values(lambda x: psi_2d_model(x, model_x, model_y))
    plot_2d_map(predicted_values, zlabel='$\psi_{{{}, {}}}$(x, y)'.format(nx, ny), output_prefix='prediction')

    plot_2d_map(np.abs(true_values - predicted_values), zlabel='|$\psi_{{{},{}, predicted}}$(x, y) - $\psi_{{{},{}, true}}$(x, y)|'.format(nx, ny, nx, ny), output_prefix='difference')

    test_metric = dde.metrics.l2_relative_error(true_values, predicted_values)    
    save_to_csv()

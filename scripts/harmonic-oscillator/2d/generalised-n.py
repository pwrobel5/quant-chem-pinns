import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 75
DEFAULT_NUM_TRAIN = 4096
DEFAULT_NUM_TEST = 10000
DEFAULT_WEIGHTS = 1

NX_MAX = 5
NY_MAX = 5
ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY_X = (NX_MAX + 1) * 2
NUM_BOUNDARY_Y = (NY_MAX + 1) * 2

LX = 2
LY = 2
m = 1
omega_x = 0.5
omega_y = 0.5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)
    parser.add_argument('-weights', '--weights', default=DEFAULT_WEIGHTS)

    return parser.parse_args()

def pde(x, y, omega):
    n = x[:, 1:2]
    E = (n + 0.5) * omega
    U = 0.5 * m * (omega ** 2) * (x[:, 0:1] ** 2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return -dy_xx / 2 + (U - E) * y

def psi(x, omega):
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

def get_collocation_points(n_max):
    points = []
    for n in range(0, n_max + 1):
        points.append((0.0, n))
    return np.array(points)

def boundary_value(x, omega):
    n = x[-1]

    if n % 2 == 1:
        return 0
    
    constants = (1.0 / (np.sqrt(sp.gamma(n + 1) * (2 ** n)))) * (((m * omega) / np.pi) ** 0.25)
    if n == 0:
        return constants
    
    hermite_coefficients = [0] * int(n) + [1]
    hermite = np.polynomial.Hermite(hermite_coefficients)
    hermite_value = hermite(0)

    return constants * hermite_value

def boundary_derivative_value(x, omega):
    n = x[-1]

    if n % 2 == 0:
        return 0
    
    constants = 2 * n * (1.0 / (np.sqrt(sp.gamma(n + 1) * (2 ** n)))) * (1.0 / (np.pi ** 0.25)) * ((m * omega) ** 0.75)
    hermite_coefficients = [0] * (int(n) - 1) + [1]
    hermite = np.polynomial.Hermite(hermite_coefficients)
    hermite_value = hermite(0)

    return constants * hermite_value

def dpsi_network(x, y, _):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    return dy_x

def save_to_csv(nx, ny, test_metric):
    row = [nx, ny, num_dense_layers, num_dense_nodes, num_train, num_test, weights, test_metric]
    
    csv_file = open('generalised-n.csv', 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()

def train_1d_nn(n_max, L, omega, num_boundary):
    domain = dde.geometry.Rectangle([-L, 0], [L, n_max])

    collocation_points = get_collocation_points(n_max)
    collocation_values = np.array([boundary_value(x, omega) for x in collocation_points])
    collocation_values = collocation_values.reshape(collocation_values.shape + (1,))
    
    derivative_values = np.array([boundary_derivative_value(x, omega) for x in collocation_points])
    derivative_values = derivative_values.reshape(derivative_values.shape + (1,))
    
    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)
    nc = dde.icbc.PointSetOperatorBC(collocation_points, derivative_values, dpsi_network)

    data = dde.data.PDE(
        domain,
        lambda x, y: pde(x, y, omega),
        [ic, nc],
        num_domain=num_train,
        num_boundary=num_boundary,
        num_test=num_test,
        solution=lambda x: psi(x, omega)
    )

    net = dde.nn.FNN(
        [2] + [num_dense_nodes] * num_dense_layers + [1],
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

    x_values_with_n = np.c_[x_values, np.ones(x_values.shape[0]) * nx]
    y_values_with_n = np.c_[y_values, np.ones(y_values.shape[0]) * ny]

    x_component = psi(x_values_with_n, omega_x)
    y_component = psi(y_values_with_n, omega_y)

    return x_component * y_component

def psi_2d_model(data, x_model, y_model, nx, ny):
    x_values = data[:, 0:1]
    y_values = data[:, 1:2]

    x_values_with_n = np.c_[x_values, np.ones(x_values.shape[0]) * nx]
    y_values_with_n = np.c_[y_values, np.ones(y_values.shape[0]) * ny]

    x_component = x_model.predict(x_values_with_n)
    y_component = y_model.predict(y_values_with_n)

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

def plot_2d_map(function_values, nx, ny, zlabel='abc', output_prefix='function'):
    plt.figure()
    im = plt.imshow(function_values, cmap=plt.cm.RdBu, extent=[-LX, LX, -LY, LY])
    plt.colorbar(im, label=zlabel)
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.savefig('{}-{}-{}-{}-{}-{}-results.png'.format(output_prefix, nx, ny, num_dense_layers, num_dense_nodes, num_train))


if __name__ == '__main__':
    args = parse_arguments()

    num_dense_layers = int(args.num_dense_layers)
    num_dense_nodes = int(args.num_dense_nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)
    weights = int(args.weights)

    model_x = train_1d_nn(NX_MAX, LX, omega_x, NUM_BOUNDARY_X)
    model_y = train_1d_nn(NY_MAX, LY, omega_y, NUM_BOUNDARY_Y)

    for nx in range(0, 7):
        for ny in range(0, 7):
            true_values = get_2d_values(lambda x: psi_2d_rectangle(x, nx, ny))
            predicted_values = get_2d_values(lambda x: psi_2d_model(x, model_x, model_y, nx, ny))
            plot_2d_map(predicted_values, nx, ny, zlabel='$\psi_{{{}, {}}}$(x, y)'.format(nx, ny), output_prefix='prediction')

            plot_2d_map(np.abs(true_values - predicted_values), nx, ny, zlabel='|$\psi_{{{},{}, predicted}}$(x, y) - $\psi_{{{},{}, true}}$(x, y)|'.format(nx, ny, nx, ny), output_prefix='difference')

            test_metric = dde.metrics.l2_relative_error(true_values, predicted_values)
            save_to_csv(nx, ny, test_metric)

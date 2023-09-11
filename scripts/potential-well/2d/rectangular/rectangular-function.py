import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 20
DEFAULT_NX = 1
DEFAULT_NY = 1
DEFAULT_NUM_TRAIN = 32
DEFAULT_NUM_TEST = 100

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']

LX = 2
LY = 2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nx', '--nx-quantum-number', default=DEFAULT_NX)
    parser.add_argument('-ny', '--ny-quantum-number', default=DEFAULT_NY)
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)

    return parser.parse_args()

def psi(x, n, L):
    k = (n * np.pi) / L
    normalization_constant = np.sqrt(2.0 / L)
    return normalization_constant * np.sin(k * (x + 0.5 * L))

def save_to_csv():
    row = [nx, ny, num_dense_layers, num_dense_nodes, num_train, num_test, test_metric]
    
    csv_file = open('rectangular-function.csv', 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()

def train_1d_nn(n, L):
    domain = dde.geometry.Interval(-L / 2, L / 2)
    data = dde.data.Function(
        domain,
        lambda x: psi(x, n, L),
        num_train,
        num_test
    )

    net = dde.nn.FNN(
        [1] + [num_dense_nodes] * num_dense_layers + [1],
        ACTIVATION,
        INITIALIZER
    )

    model = dde.Model(data, net)
    model.compile(
        OPTIMIZER,
        metrics=METRICS
    )

    model.train(iterations=ITERATIONS)
    
    return model

def psi_2d_rectangle(data, nx, ny):
    x_values = data[:, 0:1]
    y_values = data[:, 1:2]

    x_component = psi(x_values, nx, LX)
    y_component = psi(y_values, ny, LY)

    return x_component * y_component

def psi_2d_model(data, x_model, y_model):
    x_values = data[:, 0:1]
    y_values = data[:, 1:2]

    x_component = x_model.predict(x_values)
    y_component = y_model.predict(y_values)

    return x_component * y_component

def get_2d_values(input_function, grid_dist=0.02):
    x_ax = np.arange(-LX / 2, LX / 2, grid_dist)
    y_ax = np.arange(-LY / 2, LY / 2, grid_dist)
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
    im = plt.imshow(function_values, cmap=plt.cm.RdBu, extent=[-LX / 2, LX / 2, -LY / 2, LY / 2])
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

    model_x = train_1d_nn(nx, LX)
    model_y = train_1d_nn(ny, LY)

    true_values = get_2d_values(lambda x: psi_2d_rectangle(x, nx, ny))

    predicted_values = get_2d_values(lambda x: psi_2d_model(x, model_x, model_y))
    plot_2d_map(predicted_values, zlabel='$\psi_{{{}, {}}}$(x, y)'.format(nx, ny), output_prefix='prediction')

    plot_2d_map(np.abs(true_values - predicted_values), zlabel='|$\psi_{{{},{}, predicted}}$(x, y) - $\psi_{{{},{}, true}}$(x, y)|'.format(nx, ny, nx, ny), output_prefix='difference')

    test_metric = dde.metrics.l2_relative_error(true_values, predicted_values)
    save_to_csv()

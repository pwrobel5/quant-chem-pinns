import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 75
DEFAULT_NUM_TRAIN = 32
DEFAULT_NUM_TEST = 100

N_MAX = 5
ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']

L = 2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)

    return parser.parse_args()

def psi(x):
    n = x[:, 1:2]
    k = (n * np.pi) / L
    normalization_constant = np.sqrt(2.0 / L)
    return normalization_constant * np.sin(k * (x[:, 0:1] + 0.5 * L))

def predict_for_n(points):
    prediction = model.predict(points)
    return np.array(prediction)

def save_results_for_n(n, num_points=DEFAULT_NUM_TEST):
    x = np.linspace(-L / 2, L / 2, num=num_points)
    points = np.array([[i, n] for i in x])
    y_pred = predict_for_n(points)
    y_true = psi(points)

    plt.figure()
    
    plt.plot(x, y_true, 'o', color='black', label='Testing points')
    plt.plot(x, y_true, '-', color='black', label='True values')
    plt.plot(x, y_pred, '--', color='red', label='Predicted values')
    
    plt.xlabel('x')
    plt.ylabel('$\psi_{}$(x)'.format(n))
    
    plt.legend()
    plt.savefig('{}-{}-{}-{}-results.png'.format(n, num_dense_layers, num_dense_nodes, num_train))
    
    test_metric = dde.metrics.l2_relative_error(y_true, y_pred)
    row = [n, num_dense_layers, num_dense_nodes, num_train, num_test, test_metric]
    csv_file = open('linear-function-generalised-n.csv', 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()
    print('L2 relative error for n = {}: {:.2e}'.format(n, test_metric))

if __name__ == '__main__':
    args = parse_arguments()

    num_dense_layers = int(args.num_dense_layers)
    num_dense_nodes = int(args.num_dense_nodes)
    num_train = int(args.num_train) ** 2
    num_test = int(args.num_test) ** 2

    domain = dde.geometry.Rectangle([-L / 2, 1], [L / 2, N_MAX])
    data = dde.data.Function(
        domain, 
        psi, 
        num_train, 
        num_test
    )
    net = dde.nn.FNN(
        [2] + [num_dense_nodes] * num_dense_layers + [1], 
        ACTIVATION, 
        INITIALIZER
    )

    model = dde.Model(data, net)
    model.compile(
        OPTIMIZER, 
        metrics=METRICS
    )

    loss_history, train_state = model.train(iterations=ITERATIONS)
    dde.saveplot(loss_history, train_state, issave=True, isplot=False)
    
    for i in range(1, 8):
        save_results_for_n(i, num_points=int(args.num_test))
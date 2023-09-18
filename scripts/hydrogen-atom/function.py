import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 60
DEFAULT_N = 1
DEFAULT_L = 0
DEFAULT_NUM_TRAIN = 256
DEFAULT_NUM_TEST = 100
Z = 1
R_MAX = 50

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-quantum-number', default=DEFAULT_N)
    parser.add_argument('-l', '--l-quantum-number', default=DEFAULT_L)
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)

    return parser.parse_args()

def radial_part(r):
    normalization_constant = np.sqrt(
        np.power(2.0 / n, 3.0) * (np.math.factorial(n - l - 1) / (2 * n * np.math.factorial(n + l)))
    )
    
    exponent = np.exp(- (Z * r) / n)
    power = np.power(2 * Z * r / n, l)
    laguerre_polynomial = sp.genlaguerre(n - l - 1, 2 * l + 1)
    laguerre_value = laguerre_polynomial(2 * Z * r / n)
    
    return normalization_constant * exponent * power * laguerre_value

def save_prediction_plot():
    train_file = open('train.dat', 'r')
    
    x_train = []
    y_train = []
    train_file.readline()
    
    for line in train_file:
        line = line.split()
        x_train.append(float(line[0]))
        y_train.append(float(line[1]))
    
    train_file.close()
    
    test_file = open('test.dat', 'r')
    
    x_test = []
    y_true = []
    y_pred = []
    test_file.readline()
    
    for line in test_file:
        line = line.split()
        x_test.append(float(line[0]))
        y_true.append(float(line[1]))
        y_pred.append(float(line[2]))
    
    test_file.close()
    
    plt.plot(x_train, y_train, 'o', color='black', label='Training points')
    plt.plot(x_test, y_true, '-', color='black', label='True values')
    plt.plot(x_test, y_pred, '--', color='red', label='Predicted values')
    
    plt.xlabel('x')
    plt.ylabel('$R_{{{},{}}}$(x)'.format(n, l))
    
    plt.legend()
    plt.savefig('{}-{}-{}-{}-{}-results.png'.format(n, l, num_dense_layers, num_dense_nodes, num_train))

def save_to_csv():
    row = [n, l, num_dense_layers, num_dense_nodes, num_train, num_test, test_metric]
    
    csv_file = open('function.csv', 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()

if __name__ == '__main__':
    args = parse_arguments()

    n = int(args.n_quantum_number)
    l = int(args.l_quantum_number)
    num_dense_layers = int(args.num_dense_layers)
    num_dense_nodes = int(args.num_dense_nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)

    domain = dde.geometry.Interval(0, R_MAX)
    data = dde.data.Function(
        domain, 
        radial_part, 
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

    loss_history, train_state = model.train(iterations=ITERATIONS)
    dde.saveplot(loss_history, train_state, issave=True, isplot=False)
    test_metric = loss_history.metrics_test[-1][0]

    save_prediction_plot()
    save_to_csv()

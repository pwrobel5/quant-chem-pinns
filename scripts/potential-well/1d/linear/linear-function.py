import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 20
DEFAULT_N = 1
DEFAULT_NUM_TRAIN = 32
DEFAULT_NUM_TEST = 100

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']

L = 2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--quantum-number', default=DEFAULT_N)
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)

    return parser.parse_args()

def psi(x):
    k = (n * np.pi) / L
    normalization_constant = np.sqrt(2.0 / L)
    return normalization_constant * np.sin(k * (x + 0.5 * L))

def save_loss_plot():
    loss_file = open('loss.dat', 'r')
    
    loss_file.readline()
    steps = []
    train_loss = []
    test_loss = []
    test_metric = []
    
    for line in loss_file:
        line = line.split()
        steps.append(float(line[0]))
        train_loss.append(float(line[1]))
        test_loss.append(float(line[2]))
        test_metric.append(float(line[3]))
    
    loss_file.close()
    
    plt.plot(steps, train_loss, label='Train loss')
    plt.plot(steps, test_loss, label='Test loss')
    plt.plot(steps, test_metric, label='Test metric')
    
    plt.xlabel('training epoch')
    plt.yscale('log')
    
    plt.legend()
    plt.savefig('{}-loss.png'.format(n))
    #plt.show()

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
    plt.ylabel('$\psi_{}$(x)'.format(n))
    
    plt.legend()
    plt.savefig('{}-{}-{}-{}-results.png'.format(n, num_dense_layers, num_dense_nodes, num_train))

def save_to_csv():
    row = [n, num_dense_layers, num_dense_nodes, num_train, num_test, test_metric]
    
    csv_file = open('linear-function.csv', 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()

if __name__ == '__main__':
    args = parse_arguments()

    n = int(args.quantum_number)
    num_dense_layers = int(args.num_dense_layers)
    num_dense_nodes = int(args.num_dense_nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)

    domain = dde.geometry.Interval(-L / 2, L / 2)
    data = dde.data.Function(
        domain, 
        psi, 
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

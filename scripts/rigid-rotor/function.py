import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 60
DEFAULT_L = 1
DEFAULT_M = 0
DEFAULT_NUM_TRAIN = 128
DEFAULT_NUM_TEST = 100

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--l-quantum-number', default=DEFAULT_L)
    parser.add_argument('-m', '--m-quantum-number', default=DEFAULT_M)
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)

    return parser.parse_args()

def z_component(phi):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(phi * m * 1j)

def theta_part(theta):
    # 2*pi is already at the z_component
    normalization_constant = ((-1) ** m) * np.sqrt(((2 * l + 1) / 2) * (np.math.factorial(l - m) / np.math.factorial(l + m)))
    legendre_values = []
    for i in theta:
        legendre_polynomials, _ = sp.lpmn(m, l, np.cos(i[0]))
        legendre = legendre_polynomials[m, l]
        legendre_values.append([legendre])
    legendre_values = np.array(legendre_values)
    
    return normalization_constant * legendre_values

def spherical_harmonic(x):
    theta, phi = x[:, 0:1], x[:, 1:2]
    
    legendre_component = theta_part(theta)
    phi_component = z_component(phi)
    
    return legendre_component * phi_component

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
    plt.ylabel('$\Theta_{{{},{}}}$(x)'.format(l, m))
    
    plt.legend()
    plt.savefig('{}-{}-{}-{}-{}-results.png'.format(l, m, num_dense_layers, num_dense_nodes, num_train))

def save_to_csv():
    row = [l, m, num_dense_layers, num_dense_nodes, num_train, num_test, test_metric]
    
    csv_file = open('function.csv', 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()

if __name__ == '__main__':
    args = parse_arguments()

    l = int(args.l_quantum_number)
    m = int(args.m_quantum_number)
    num_dense_layers = int(args.num_dense_layers)
    num_dense_nodes = int(args.num_dense_nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)

    domain = dde.geometry.Interval(0, np.pi)
    data = dde.data.Function(
        domain, 
        theta_part, 
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

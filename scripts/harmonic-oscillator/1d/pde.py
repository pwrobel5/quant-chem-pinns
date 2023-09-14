import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 20
DEFAULT_N = 0
DEFAULT_NUM_TRAIN = 64
DEFAULT_NUM_TEST = 100
DEFAULT_WEIGHTS = 1

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY = 2

L = 5
m = 1
omega = 0.5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--quantum-number', default=DEFAULT_N)
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)
    parser.add_argument('-weights', '--weights', default=DEFAULT_WEIGHTS)

    return parser.parse_args()

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    E = (n + 0.5) * omega
    U = 0.5 * m * (omega ** 2) * (x ** 2)
    return -dy_xx / 2 + (U - E) * y

def psi(x):
    constants = (1.0 / (np.sqrt(np.math.factorial(n) * (2 ** n)))) * (((m * omega) / np.pi) ** 0.25)
    exponent = np.exp(-0.5 * m * omega * np.power(x, 2))
    hermite_coefficients = [0] * n + [1]
    hermite = np.polynomial.hermite.Hermite(hermite_coefficients)
    hermite_value = hermite(x * np.sqrt(m * omega))
    result = constants * exponent * hermite_value
    
    return result

def x_boundary_even(x, _):
    return np.isclose(x[0], 0)

def x_boundary_odd(x, _):
    return np.isclose(x[0], L / 2) or np.isclose(x[0], -L / 2)

def boundary_value(_):
    if n % 2 == 1:
        return 0
    
    constants = (1.0 / (np.sqrt(np.math.factorial(n) * (2 ** n)))) * (((m * omega) / np.pi) ** 0.25)
    if n == 0:
        return constants
    
    hermite_coefficients = [0] * n + [1]
    hermite = np.polynomial.Hermite(hermite_coefficients)
    hermite_value = hermite(0)

    return constants * hermite_value

def boundary_derivative_value(_):
    if n % 2 == 0:
        return 0
    
    constants = 2 * n * (1.0 / (np.sqrt(np.math.factorial(n) * (2 ** n)))) * (1.0 / (np.pi ** 0.25)) * ((m * omega) ** 0.75)
    hermite_coefficients = [0] * (n - 1) + [1]
    hermite = np.polynomial.Hermite(hermite_coefficients)
    hermite_value = hermite(0)

    return constants * hermite_value

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
    
    x_true, y_true = zip(*sorted(zip(x_test, y_true)))
    x_pred, y_pred = zip(*sorted(zip(x_test, y_pred)))
    
    plt.plot(x_train, y_train, 'o', color='black', label='Training points')
    plt.plot(x_true, y_true, '-', color='black', label='True values')
    plt.plot(x_pred, y_pred, '--', color='red', label='Predicted values')
    
    plt.xlabel('x')
    plt.ylabel('$\psi_{}$(x)'.format(n))
    
    plt.legend()
    plt.savefig('{}-{}-{}-{}-{}-results.png'.format(n, num_dense_layers, num_dense_nodes, num_train, weights))

def save_to_csv():
    row = [n, num_dense_layers, num_dense_nodes, num_train, num_test, weights, test_metric]
    
    csv_file = open('pde.csv', 'a')

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
    weights = int(args.weights)

    domain = dde.geometry.Interval(-L, L)

    if n % 2 == 0:
        dirichlet_bc = dde.icbc.DirichletBC(domain, boundary_value, x_boundary_even)
    else:
        dirichlet_bc = dde.icbc.DirichletBC(domain, psi, x_boundary_odd)
    neumann_bc = dde.icbc.NeumannBC(domain, boundary_derivative_value, x_boundary_even)

    data = dde.data.PDE(
        domain, 
        pde, 
        [dirichlet_bc, neumann_bc], 
        num_domain=num_train, 
        num_boundary=NUM_BOUNDARY,
        solution=psi, 
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

    loss_history, train_state = model.train(iterations=ITERATIONS)
    dde.saveplot(loss_history, train_state, issave=True, isplot=False)
    test_metric = loss_history.metrics_test[-1][0]

    save_prediction_plot()
    save_to_csv()

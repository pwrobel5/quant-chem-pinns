import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.special as sp
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 75
DEFAULT_NUM_TRAIN = 32
DEFAULT_NUM_TEST = 100
DEFAULT_WEIGHTS = 1

N_MAX = 5
ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY = (N_MAX + 1) * 2

L = 5
m = 1
omega = 0.5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)
    parser.add_argument('-weights', '--weights', default=DEFAULT_WEIGHTS)

    return parser.parse_args()

def pde(x, y):
    n = x[:, 1:2]
    E = (n + 0.5) * omega
    U = 0.5 * m * (omega ** 2) * (x[:, 0:1] ** 2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return -dy_xx / 2 + (U - E) * y

def psi(x):
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

def boundary_value(x):
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

def boundary_derivative_value(x):
    n = x[-1]

    if n % 2 == 0:
        return 0
    
    constants = 2 * n * (1.0 / (np.sqrt(sp.gamma(n + 1) * (2 ** n)))) * (1.0 / (np.pi ** 0.25)) * ((m * omega) ** 0.75)
    hermite_coefficients = [0] * (int(n) - 1) + [1]
    hermite = np.polynomial.Hermite(hermite_coefficients)
    hermite_value = hermite(0)

    return constants * hermite_value

def predict_for_n(points):
    prediction = model.predict(points)
    return np.array(prediction)

def dpsi_network(x, y, _):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    return dy_x

def save_results_for_n(n, num_points=DEFAULT_NUM_TEST):
    x = np.linspace(-L, L, num=num_points)
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
    plt.savefig('{}-{}-{}-{}-{}-results.png'.format(n, num_dense_layers, num_dense_nodes, num_train, weights))
    
    test_metric = dde.metrics.l2_relative_error(y_true, y_pred)
    row = [n, num_dense_layers, num_dense_nodes, num_train, num_test, weights, test_metric]
    csv_file = open('generalised-n.csv', 'a')

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
    weights = int(args.weights)

    domain = dde.geometry.Rectangle([-L, 0], [L, N_MAX])

    collocation_points = get_collocation_points(N_MAX)
    collocation_values = np.array([boundary_value(x) for x in collocation_points])
    collocation_values = collocation_values.reshape(collocation_values.shape + (1,))

    derivative_values = np.array([boundary_derivative_value(x) for x in collocation_points])
    derivative_values = derivative_values.reshape(derivative_values.shape + (1,))

    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)
    nc = dde.icbc.PointSetOperatorBC(collocation_points, derivative_values, dpsi_network)

    data = dde.data.PDE(
        domain, 
        pde, 
        [ic, nc], 
        num_domain=num_train,
        num_boundary=NUM_BOUNDARY,
        solution=psi, 
        num_test=num_test
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

    loss_history, train_state = model.train(iterations=ITERATIONS)
    dde.saveplot(loss_history, train_state, issave=True, isplot=False)

    for i in range(0, 7):
        save_results_for_n(i, num_points=int(args.num_test))

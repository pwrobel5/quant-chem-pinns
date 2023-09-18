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
DEFAULT_WEIGHTS = 1

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY = 2

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

def pde(x, y):
    dy_r = dde.grad.jacobian(y, x, i=0, j=0)
    dy_rr = dde.grad.hessian(y, x, i=0, j=0)
    
    E = (Z ** 2) / (2 * (n ** 2))
    hessian_part = - ((x ** 2) / 2) * dy_rr
    jacobian_part = - x * dy_r
    y_part = 0.5 * l * (l + 1) * y - Z * x * y + E * (x ** 2) * y
    
    return hessian_part + jacobian_part + y_part

def get_collocation_points(n, l):
    r_values = []
    
    r_interval = R_MAX / (4 * (n + l))
    
    for k in range(3 * (n + l) - 1):
        r = (k + 1) * r_interval
        r_values.append(r)
    
    return np.array(r_values).reshape((3 * (n + l) - 1, 1))

def boundary(x, on_boundary):
    return on_boundary and np.isclose(x, R_MAX)[0]

def boundary_value(x):
    return 0

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
    plt.ylabel('$R_{{{},{}}}$(x)'.format(n, l))
    
    plt.legend()
    plt.savefig('{}-{}-{}-{}-{}-{}-results.png'.format(n, l, num_dense_layers, num_dense_nodes, num_train, weights))

def save_to_csv():
    row = [n, l, num_dense_layers, num_dense_nodes, num_train, num_test, weights, test_metric]
    
    csv_file = open('pde.csv', 'a')

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
    weights = DEFAULT_WEIGHTS

    domain = dde.geometry.Interval(0, R_MAX)

    collocation_points = get_collocation_points(n, l)
    collocation_values = radial_part(collocation_points)
    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)
    dirichlet_bc = dde.icbc.DirichletBC(domain, boundary_value, boundary)

    data = dde.data.PDE(
        domain, 
        pde, 
        [ic, dirichlet_bc],
        num_domain=num_train, 
        num_boundary=NUM_BOUNDARY,
        solution=radial_part, 
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

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
DEFAULT_NUM_TRAIN = 64
DEFAULT_NUM_TEST = 100
DEFAULT_WEIGHTS = 1

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY = 2

R = 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-quantum-number', default=DEFAULT_N)
    parser.add_argument('-l', '--l-quantum-number', default=DEFAULT_L)
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)
    parser.add_argument('-weights', '--weights', default=DEFAULT_WEIGHTS)

    return parser.parse_args()

def pde(x, y, n, l):
    dy_r = dde.grad.jacobian(y, x, i=0, j=0)
    dy_rr = dde.grad.hessian(y, x, i=0, j=0)
    
    k = sp.jn_zeros(l, n)[-1] 
    
    return (x ** 2) * dy_rr + x * dy_r + ((x ** 2) * (k ** 2) - (l ** 2)) * y

def angular(theta, l):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(theta * l * 1j)

def radial(r, n, l):
    alpha = sp.jn_zeros(l, n)[-1]
    normalization_constant = np.sqrt(2) / (R * np.abs(sp.jv(l + 1, alpha)))
    bessel = sp.jv(l, alpha * (r / R))

    return normalization_constant * bessel

def psi(x, n, l):
    r, theta = x[:, 0:1], x[:, 1:2]
    
    radial_part = radial(r, n, l)
    angular_part = angular(theta)

    return radial_part * angular_part

def get_collocation_points(n, l):
    rs = []
    
    r_interval = R / (3 * (n + l))
    
    for k in range(3 * (n + l) - 1):
        r = (k + 1) * r_interval
        rs.append(r)
    
    return np.array(rs).reshape((3 * (n + l) - 1, 1))

def boundary(x, on_boundary):
    return on_boundary

def boundary_value(x, n, l):
    return 0

def boundary_derivative_value(x, n, l):
    alpha = sp.jn_zeros(l, n)[-1]
    normalization_constant = (np.sqrt(2) * alpha) / ((R ** 2) * np.abs(sp.jv(l + 1, alpha)))
    return normalization_constant * sp.jvp(l, alpha * x / R)

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
    plt.ylabel('$\psi_{{{},{}}}$(x)'.format(n, l))
    
    plt.legend()
    plt.savefig('{}-{}-{}-{}-{}-{}-results.png'.format(n, l, num_dense_layers, num_dense_nodes, num_train, weights))

def save_to_csv():
    row = [n, l, num_dense_layers, num_dense_nodes, num_train, num_test, weights, test_metric]
    
    csv_file = open('circular-pde.csv', 'a')

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
    weights = int(args.weights)

    domain = dde.geometry.Interval(0, R)

    collocation_points = get_collocation_points(n, l)
    collocation_values = radial(collocation_points, n, l)
    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)

    dirichlet_bc = dde.icbc.DirichletBC(domain, lambda x: boundary_value(x, n, l), boundary)
    neumann_bc = dde.icbc.NeumannBC(domain, lambda x: boundary_derivative_value(x, n, l), boundary)

    bcs = [ic, dirichlet_bc, neumann_bc]

    singularity = [0.0]

    data = dde.data.PDE(
        domain, 
        lambda x, y: pde(x, y, n, l), 
        bcs,
        num_domain=num_train, 
        num_boundary=NUM_BOUNDARY,
        solution=lambda x: radial(x, n, l), 
        num_test=num_test,
        exclusions=singularity
    )
    net = dde.nn.FNN(
        [1] + [num_dense_nodes] * num_dense_layers + [1], 
        ACTIVATION, 
        INITIALIZER
    )

    model = dde.Model(data, net)
    loss_weights = [1, weights, weights, weights]
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

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
DEFAULT_WEIGHTS = 100

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY = 2

L = 2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--quantum-number', default=DEFAULT_N)
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)
    parser.add_argument('-weights', '--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('-random', '--random-collocation-points', action='store_true')

    return parser.parse_args()

def pde(x, y):
    k = (n * np.pi) / L
    E = 0.5 * (k ** 2)
    dy_xx = dde.grad.hessian(y, x)
    return 0.5 * dy_xx + E * y

def psi(x):
    k = (n * np.pi) / L
    normalization_constant = np.sqrt(2.0 / L)
    return normalization_constant * np.sin(k * (x + 0.5 * L))

def get_extremal_collocation_points(n):
    points = []
    for k in range(n):
        x = (k * L) / n + L / (2 * n) - L / 2
        points.append([x])
        
    if n == 1:
        points.append([L / 4])
        points.append([-L / 4])
    
    return np.array(points)

def get_random_collocation_points(n):
    points = []
    for _ in range(n):
        x = np.random.uniform(low=-L/2, high=L/2)
        points.append([x])
    
    if n == 1:
        points.append([np.random.uniform(low=-L/2, high=L/2)])
        points.append([np.random.uniform(low=-L/2, high=L/2)])
    
    return np.array(points)

def x_boundary(_, on_boundary):
    return on_boundary

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
    plt.savefig('{}-{}-{}-{}-{}-{}-results.png'.format(n, num_dense_layers, num_dense_nodes, num_train, is_random, weights))

def save_to_csv():
    row = [n, num_dense_layers, num_dense_nodes, num_train, num_test, is_random, weights, test_metric]
    
    csv_file = open('linear-pde.csv', 'a')

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
    is_random = True if args.random_collocation_points else False

    domain = dde.geometry.Interval(-L / 2, L / 2)

    collocation_points = get_random_collocation_points(n) if is_random else get_extremal_collocation_points(n)
    collocation_values = psi(collocation_points)
    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)

    dirichlet_bc = dde.icbc.DirichletBC(domain, lambda x: 0, x_boundary)

    data = dde.data.PDE(
        domain, 
        pde, 
        [ic, dirichlet_bc], 
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

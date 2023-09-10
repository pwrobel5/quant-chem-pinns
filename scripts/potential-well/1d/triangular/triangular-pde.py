import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 20
DEFAULT_N = 1
DEFAULT_NUM_TRAIN = 32
DEFAULT_NUM_TEST = 100
DEFAULT_WEIGHTS = 10

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY = 2

R = 2
q = 1
epsilon = 1


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
    ai_zeros, _, _, _ = sp.ai_zeros(n)
    alpha_n = ai_zeros[-1]
    E = -alpha_n * np.power((q ** 2) * (epsilon ** 2) / 2, (1 / 3))
    
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    
    return dy_xx + 2 * (E - q * epsilon * x) * y

def psi(x):
    ai_zeros, _, _, _ = sp.ai_zeros(n)
    alpha_n = ai_zeros[-1]
    E = -alpha_n * np.power((q ** 2) * (epsilon ** 2) / 2, (1 / 3))
    
    cube_root = np.power(2 * q * epsilon, (1 / 3))
    
    ai_value, _, _, _ = sp.airy(cube_root * (x - (E / (q * epsilon))))
    _, aip_value, _, _ = sp.airy(-cube_root * (E / (q * epsilon)))
    
    normalization_constant = np.sqrt(cube_root / (aip_value ** 2))
    
    return normalization_constant * ai_value

def get_collocation_points(n):
    rs = []
    
    r_interval = R / (3 * n)
    
    for k in range(3 * n - 1):
        r = (k + 1) * r_interval
        rs.append(r)
    
    return np.array(rs).reshape((3 * n - 1, 1))

def boundary(x, on_boundary):
    return on_boundary and np.isclose(0, x[0])

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
    plt.ylabel('$\psi_{}$(x)'.format(n))
    
    plt.legend()
    plt.savefig('{}-{}-{}-{}-{}-results.png'.format(n, num_dense_layers, num_dense_nodes, num_train, weights))

def save_to_csv():
    row = [n, num_dense_layers, num_dense_nodes, num_train, num_test, weights, test_metric]
    
    csv_file = open('triangular-pde.csv', 'a')

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

    domain = dde.geometry.Interval(0, R)

    collocation_points = get_collocation_points(n)
    collocation_values = psi(collocation_points)
    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)

    dirichlet_bc = dde.icbc.DirichletBC(domain, boundary_value, boundary)

    data = dde.data.PDE(
        domain, 
        pde, 
        [ic, dirichlet_bc], 
        num_domain=num_train, 
        num_boundary=NUM_BOUNDARY,
        solution=psi, 
        num_test=num_test,
        exclusions=[R]
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

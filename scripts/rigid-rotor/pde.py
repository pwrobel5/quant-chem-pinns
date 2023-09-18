import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import tensorflow.math as tfmath
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 60
DEFAULT_L = 1
DEFAULT_M = 0
DEFAULT_NUM_TRAIN = 128
DEFAULT_NUM_TEST = 100
DEFAULT_WEIGHTS = 1

ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY = 2

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

def pde(x, y):
    dy_theta = dde.grad.jacobian(y, x, i=0, j=0)
    dy_thetatheta = dde.grad.hessian(y, x, i=0, j=0)
    
    cos_theta = tfmath.cos(x)
    sin_theta = tfmath.sin(x)
    
    # be aware that sometimes it is better to multiply both sides to avoid points with division by 0, i.e. here,
    # to multiply both sides by sin^2 theta, as in normal form of this equation it divides m^2 what led to
    # NaNs during training process
    return (sin_theta ** 2) * dy_thetatheta + (cos_theta * sin_theta) * dy_theta + ((sin_theta ** 2) * l * (l + 1) - (m ** 2)) * y

def get_collocation_points(l, m):
    theta_values = []
    
    theta_interval = np.pi / (3 * (l + m + 1))
    
    for k in range(3 * (l + m + 1) -1):
        theta = (k + 1) * theta_interval
        theta_values.append(theta)
    
    return np.array(theta_values).reshape((3 * (l + m + 1) -1, 1))

def boundary(x, on_boundary):
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
    plt.ylabel('$\Theta_{{{},{}}}$(x)'.format(l, m))
    
    plt.legend()
    plt.savefig('{}-{}-{}-{}-{}-{}-results.png'.format(l, m, num_dense_layers, num_dense_nodes, num_train, weights))

def save_to_csv():
    row = [l, m, num_dense_layers, num_dense_nodes, num_train, num_test, weights, test_metric]
    
    csv_file = open('pde.csv', 'a')

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
    weights = DEFAULT_WEIGHTS

    domain = dde.geometry.Interval(0, np.pi)

    collocation_points = get_collocation_points(l, m)
    collocation_values = theta_part(collocation_points)
    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)

    data = dde.data.PDE(
        domain, 
        pde, 
        ic,
        num_domain=num_train, 
        num_boundary=NUM_BOUNDARY,
        solution=theta_part, 
        num_test=num_test
    )

    net = dde.nn.FNN(
        [1] + [num_dense_nodes] * num_dense_layers + [1], 
        ACTIVATION, 
        INITIALIZER
    )

    model = dde.Model(data, net)
    loss_weights = [1, weights]
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

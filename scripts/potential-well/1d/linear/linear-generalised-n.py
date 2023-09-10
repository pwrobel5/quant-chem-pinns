import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 75
DEFAULT_NUM_TRAIN = 32
DEFAULT_NUM_TEST = 100
DEFAULT_WEIGHTS = 10

N_MAX = 5
ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY = N_MAX * 2

L = 2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ndense', '--num-dense-layers', default=DEFAULT_DENSE_LAYERS)
    parser.add_argument('-nnodes', '--num-dense-nodes', default=DEFAULT_DENSE_NODES)
    parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
    parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)
    parser.add_argument('-weights', '--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('-random', '--random-collocation-points', action='store_true')

    return parser.parse_args()

def pde(x, y):
    n = x[:, 1:2]
    k = (n * np.pi) / L
    E = 0.5 * (k ** 2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return 0.5 * dy_xx + E * y

def psi(x):
    normalization_constant = np.sqrt(2.0 / L)
    n = x[:, 1:2]
    k = (n * np.pi) / L
    return normalization_constant * np.sin(k * (x[:, 0:1] + (L / 2)))

def get_extremal_collocation_points(n_max):
    points = []
    for n in range(1, n_max + 1):
        for k in range(n):
            x = (k * L) / n + L / (2 * n) - L / 2
            points.append((x, n))
            
        if n == 1:
            points.append((L / 4, n))
            points.append((-L / 4, n))
    
    return np.array(points)

def get_random_collocation_points(n_max):
    points = []
    for n in range(1, n_max + 1):
        for _ in range(n):
            x = np.random.uniform(low=-L/2, high=L/2)
            points.append((x, n))
        
        if n == 1:
            points.append((np.random.uniform(low=-L/2, high=L/2), n))
            points.append((np.random.uniform(low=-L/2, high=L/2), n))
    
    return np.array(points)

def x_boundary(x, on_boundary):
    if x[1].is_integer():
        return np.isclose(-L / 2, x[0]) or np.isclose(L / 2, x[0])
    return False

def predict_for_n(points):
    prediction = model.predict(points)
    return np.array(prediction)

def save_results_for_n(n, num_points=DEFAULT_NUM_TEST):
    x = np.linspace(-L / 2, L / 2, num=num_points)
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
    plt.savefig('{}-{}-{}-{}-{}-{}-results.png'.format(n, num_dense_layers, num_dense_nodes, num_train, is_random, weights))
    
    test_metric = dde.metrics.l2_relative_error(y_true, y_pred)
    row = [n, num_dense_layers, num_dense_nodes, num_train, num_test, is_random, weights, test_metric]
    csv_file = open('linear-generalised-n.csv', 'a')

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
    is_random = True if args.random_collocation_points else False

    domain = dde.geometry.Rectangle([-L / 2, 1], [L / 2, N_MAX])

    collocation_points = get_random_collocation_points(N_MAX) if is_random else get_extremal_collocation_points(N_MAX)
    collocation_values = psi(collocation_points)
    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)

    bc = dde.icbc.DirichletBC(domain, lambda x: 0, x_boundary)

    data = dde.data.PDE(
        domain, 
        pde, 
        [ic, bc], 
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

    for i in range(1, 8):
        save_results_for_n(i, num_points=int(args.num_test))

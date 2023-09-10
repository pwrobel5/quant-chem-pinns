import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import random as rnd

from deepxde.geometry.sampler import sample

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


class LogUniformPoints(Exception):
    pass

class UniformBoundaryPoints(Exception):
    pass

class PeriodicPoint(Exception):
    pass

class BackgroundPoints(Exception):
    pass

class Quantum1D(dde.geometry.geometry_1d.Interval):
    def __init__(self, x_min, x_max, n_min, n_max):
        super().__init__(x_min, x_max)
        self._n_min = n_min
        self._n_max = n_max
    
    def inside(self, x):
        print('inside')
        x_inside = np.logical_and(self.l <= x[:, 0:1], x[:, 0:1] <= self.r).flatten()
        n_integer = np.array([n[0].is_integer for n in x[:, 1:2]]).reshape(x_inside.shape)
        return np.logical_and(x_inside, n_integer)
    
    def on_boundary(self, x):
        print('on_boundary')
        return super().on_boundary(x[:, 0:1])
    
    def distance2boundary(self, x, dirn):
        print('distance2boundary')
        return super().distance2boundary(x[:, 0:1], dirn)
    
    def mindist2boundary(self, x):
        print('mindist2boundary')
        return super().mindist2boundary(x[:, 0:1])
    
    def boundary_normal(self, x):
        print('boundary_normal')
        return super().boundary_normal(x[:, 0:1])
    
    def uniform_points(self, n, boundary=True):
        print('uniform_points')
        #raise UniformPoints
        
        xs_per_n = n // self._n_max
        xs = super().uniform_points(xs_per_n, boundary=boundary)
        uniform_points = []
        
        for i in range(1, self._n_max + 1):
            for x in xs:
                uniform_points.append(np.hstack([x, i]))
        rest = n - (xs_per_n * self._n_max)
        if rest != 0:
            print('WARNING: non zero rest points: {}'.format(rest))
               
        return np.array(uniform_points)
    
    def log_uniform_points(self, n, boundary=True):
        print('log_uniform_points')
        raise LogUniformPoints
        return super().log_uniform_points(n)
    
    def random_points(self, n, random='pseudo'):
        #xs_per_n = n // self._n_max
        random_points = []
        
        xs = sample(n, 1, random)
        xs = self.diam * xs + self.l
        for x in xs:
            i = np.random.randint(1, self._n_max + 1)
            random_points.append(np.hstack([x, i]))
              
        print('random_points')
        return np.array(random_points)
    
    def uniform_boundary_points(self, n):
        print('uniform_boundary_points')
        raise UniformBoundaryPoints
        return super().uniform_boundary_points(n)
    
    def random_boundary_points(self, n, random='pseudo'):
        print('random_boundary_points')
        
        xs_per_n = n // self._n_max
        random_boundary_points = []
        
        for i in range(1, self._n_max + 1):
            if xs_per_n == 2:
                random_boundary_points.append([self.l, i])
                random_boundary_points.append([self.r, i])
            else:
                xs = np.random.choice([self.l, self.r], xs_per_n)
                for x in xs:
                    random_boundary_points.append([x, i])
        
        rest = n - (xs_per_n * self._n_max)
        if rest != 0:
            print('WARNING: non zero rest points: {}'.format(rest))
        
        return np.array(random_boundary_points)
    
    def periodic_point(self, x, component=0):
        print('periodic_point')
        raise PeriodicPoint
        return super().periodic_point(x)
    
    def background_points(self, x, dirn, dist2npt, shift):
        print('background_points')
        raise BackgroundPoints
        return super().background_points(self, x, dirn, dist2npt, shift)


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
    csv_file = open('linear-quantum-1d.csv', 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()
    print('L2 relative error for n = {}: {:.2e}'.format(n, test_metric))


if __name__ == '__main__':
    args = parse_arguments()

    num_dense_layers = int(args.num_dense_layers)
    num_dense_nodes = int(args.num_dense_nodes)
    num_train = int(args.num_train) * N_MAX
    num_test = int(args.num_test) * N_MAX
    weights = int(args.weights)
    is_random = True if args.random_collocation_points else False

    domain = Quantum1D(-L /2, L/2, 1, N_MAX)

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

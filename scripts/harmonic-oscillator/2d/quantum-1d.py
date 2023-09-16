import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import random as rnd

from deepxde.geometry.sampler import sample

dde.config.set_random_seed(1234)

DEFAULT_DENSE_LAYERS = 5
DEFAULT_DENSE_NODES = 60
DEFAULT_NUM_TRAIN = 384
DEFAULT_NUM_TEST = 600
DEFAULT_WEIGHTS = 1

NX_MAX = 5
NY_MAX = 5
ITERATIONS = 10000
ACTIVATION = 'tanh'
INITIALIZER = 'Glorot uniform'
OPTIMIZER = 'L-BFGS'
METRICS = ['l2 relative error']
NUM_BOUNDARY_X = (NX_MAX + 1) * 2
NUM_BOUNDARY_Y = (NY_MAX + 1) * 2

LX = 5
LY = 5
m = 1
omega_x = 0.5
omega_y = 0.5


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
        
        xs_per_n = n // (self._n_max + 1)
        xs = super().uniform_points(xs_per_n, boundary=boundary)
        uniform_points = []
        
        for i in range(self._n_min, self._n_max + 1):
            for x in xs:
                uniform_points.append(np.hstack([x, i]))
        rest = n - (xs_per_n * (self._n_max + 1))
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
            i = np.random.randint(self._n_min, self._n_max + 1)
            random_points.append(np.hstack([x, i]))
        
        '''for i in range(1, self._n_max + 1):
            xs = sample(xs_per_n, 1, random)
            xs = self.diam * xs + self.l
            for x in xs:
                random_points.append(np.hstack([x, i]))'''
        
        '''rest = n - (xs_per_n * self._n_max)
        if rest != 0:
            print('WARNING: non zero rest points: {}'.format(rest))'''
        
        print('random_points')
        return np.array(random_points)
    
    def uniform_boundary_points(self, n):
        print('uniform_boundary_points')
        raise UniformBoundaryPoints
        return super().uniform_boundary_points(n)
    
    def random_boundary_points(self, n, random='pseudo'):
        print('random_boundary_points')
        
        xs_per_n = n // (self._n_max + 1)
        random_boundary_points = []
        
        for i in range(self._n_min, self._n_max + 1):
            if xs_per_n == 2:
                random_boundary_points.append([self.l, i])
                random_boundary_points.append([self.r, i])
            else:
                xs = np.random.choice([self.l, self.r], xs_per_n)
                for x in xs:
                    random_boundary_points.append([x, i])
        
        rest = n - (xs_per_n * (self._n_max + 1))
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

    return parser.parse_args()

def pde(x, y, omega):
    n = x[:, 1:2]
    E = (n + 0.5) * omega
    U = 0.5 * m * (omega ** 2) * (x[:, 0:1] ** 2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return -dy_xx / 2 + (U - E) * y

def psi(x, omega):
    n = x[:, 1:2]
    factorials = []
    for n_val in n:
        factorials.append(np.math.factorial(int(n_val)))
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

def boundary_value(x, omega):
    n = int(x[-1])

    if n % 2 == 1:
        return 0
    
    constants = (1.0 / (np.sqrt(np.math.factorial(n) * (2 ** n)))) * (((m * omega) / np.pi) ** 0.25)
    if n == 0:
        return constants
    
    hermite_coefficients = [0] * n + [1]
    hermite = np.polynomial.Hermite(hermite_coefficients)
    hermite_value = hermite(0)

    return constants * hermite_value

def boundary_derivative_value(x, omega):
    n = int(x[-1])

    if n % 2 == 0:
        return 0
    
    constants = 2 * n * (1.0 / (np.sqrt(np.math.factorial(n) * (2 ** n)))) * (1.0 / (np.pi ** 0.25)) * ((m * omega) ** 0.75)
    hermite_coefficients = [0] * (n - 1) + [1]
    hermite = np.polynomial.Hermite(hermite_coefficients)
    hermite_value = hermite(0)

    return constants * hermite_value

def dpsi_network(x, y, _):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    return dy_x

def save_to_csv(nx, ny, test_metric):
    row = [nx, ny, num_dense_layers, num_dense_nodes, num_train, num_test, weights, test_metric]
    
    csv_file = open('quantum-1d.csv', 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()

def train_1d_nn(n_max, L, omega, num_boundary):
    domain = Quantum1D(-L, L, 0, n_max)

    collocation_points = get_collocation_points(n_max)
    collocation_values = np.array([boundary_value(x, omega) for x in collocation_points])
    collocation_values = collocation_values.reshape(collocation_values.shape + (1,))

    derivative_values = np.array([boundary_derivative_value(x, omega) for x in collocation_points])
    derivative_values = derivative_values.reshape(derivative_values.shape + (1,))

    ic = dde.icbc.PointSetBC(collocation_points, collocation_values)
    nc = dde.icbc.PointSetOperatorBC(collocation_points, derivative_values, dpsi_network)

    data = dde.data.PDE(
        domain,
        lambda x, y: pde(x, y, omega),
        [ic, nc],
        num_domain=num_train,
        num_boundary=num_boundary,
        num_test=num_test,
        solution=lambda x: psi(x, omega)
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

    model.train(iterations=ITERATIONS)

    return model

def psi_2d_rectangle(data, nx, ny):
    x_values = data[:, 0:1]
    y_values = data[:, 1:2]

    x_values_with_n = np.c_[x_values, np.ones(x_values.shape[0]) * nx]
    y_values_with_n = np.c_[y_values, np.ones(y_values.shape[0]) * ny]

    x_component = psi(x_values_with_n, LX)
    y_component = psi(y_values_with_n, LY)

    return x_component * y_component

def psi_2d_model(data, x_model, y_model, nx, ny):
    x_values = data[:, 0:1]
    y_values = data[:, 1:2]

    x_values_with_n = np.c_[x_values, np.ones(x_values.shape[0]) * nx]
    y_values_with_n = np.c_[y_values, np.ones(y_values.shape[0]) * ny]

    x_component = x_model.predict(x_values_with_n)
    y_component = y_model.predict(y_values_with_n)

    return x_component * y_component

def get_2d_values(input_function, grid_dist=0.02):
    x_ax = np.arange(-LX, LX, grid_dist)
    y_ax = np.arange(-LY, LY , grid_dist)
    grid_x, grid_y = np.meshgrid(x_ax, y_ax)
    
    function_list = []
    for a, b in zip(grid_x, grid_y):
        pair_list = []
        for x, y in zip(a, b):
            pair_list.append([x, y])
        pairs = np.array(pair_list)
        function_list.append(np.squeeze(input_function(pairs)))
    
    function_values = np.array(function_list)
    return function_values

def plot_2d_map(function_values, nx, ny, zlabel='abc', output_prefix='function'):
    plt.figure()
    im = plt.imshow(function_values, cmap=plt.cm.RdBu, extent=[-LX, LX, -LY, LY])
    plt.colorbar(im, label=zlabel)
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.savefig('{}-{}-{}-{}-{}-{}-results.png'.format(output_prefix, nx, ny, num_dense_layers, num_dense_nodes, num_train))
    plt.close()


if __name__ == '__main__':
    args = parse_arguments()

    num_dense_layers = int(args.num_dense_layers)
    num_dense_nodes = int(args.num_dense_nodes)
    num_train = int(args.num_train)
    num_test = int(args.num_test)
    weights = int(args.weights)

    model_x = train_1d_nn(NX_MAX, LX, omega_x, NUM_BOUNDARY_X)
    model_y = train_1d_nn(NY_MAX, LY, omega_y, NUM_BOUNDARY_Y)

    for nx in range(0, 7):
        for ny in range(0, 7):
            true_values = get_2d_values(lambda x: psi_2d_rectangle(x, nx, ny))
            predicted_values = get_2d_values(lambda x: psi_2d_model(x, model_x, model_y, nx, ny))
            plot_2d_map(predicted_values, nx, ny, zlabel='$\psi_{{{}, {}}}$(x, y)'.format(nx, ny), output_prefix='prediction')

            plot_2d_map(np.abs(true_values - predicted_values), nx, ny, zlabel='|$\psi_{{{},{}, predicted}}$(x, y) - $\psi_{{{},{}, true}}$(x, y)|'.format(nx, ny, nx, ny), output_prefix='difference')

            test_metric = dde.metrics.l2_relative_error(true_values, predicted_values)
            save_to_csv(nx, ny, test_metric)

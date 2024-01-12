import csv
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

from ..utils import get_values_for_n
from .function.continuousn import FunctionContinuousN
from .pde.pde import PDEApproach


def sort_xy_data(x: list[float], y: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(x)
    y = np.array(y)

    sorted_indices = x.argsort()
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    return x_sorted, y_sorted

def save_loss_plot(plot_file_name: str) -> None:
    loss_file = open('loss.dat', 'r')
    
    loss_file.readline()
    steps = []
    train_loss = []
    test_loss = []
    test_metric = []
    
    for line in loss_file:
        line = line.split()
        steps.append(float(line[0]))
        train_loss.append(float(line[1]))
        test_loss.append(float(line[2]))
        test_metric.append(float(line[3]))
    
    loss_file.close()

    plt.figure()    
    plt.plot(steps, train_loss, label='Train loss')
    plt.plot(steps, test_loss, label='Test loss')
    plt.plot(steps, test_metric, label='Test metric')
    
    plt.xlabel('training epoch')
    plt.yscale('log')
    
    plt.legend()
    plt.savefig('{}-loss.png'.format(plot_file_name))


def save_prediction_plot(function_name: str, plot_file_name: str) -> None:
    train_file = open('train.dat', 'r')
    
    x_train = []
    y_train = []
    train_file.readline()
    
    for line in train_file:
        line = line.split()
        x_train.append(float(line[0]))
        y_train.append(float(line[1]))
    
    train_file.close()

    x_train, y_train = sort_xy_data(x_train, y_train)
    
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

    x_test_sorted, y_true = sort_xy_data(x_test, y_true)
    x_test_sorted, y_pred = sort_xy_data(x_test, y_pred)
    
    save_prediction_plot_from_points(function_name, plot_file_name, x_train, x_test_sorted, y_train, y_true, y_pred)

def save_prediction_plot_from_points(function_name: str, plot_file_name: str, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, train_label: str = 'Training points') -> None:
    plt.figure()
    plt.plot(x_train, y_train, 'o', color='black', label=train_label)
    plt.plot(x_test, y_true, '-', color='black', label='True values')
    plt.plot(x_test, y_pred, '--', color='red', label='Predicted values')
    
    plt.xlabel('x')
    plt.ylabel('{}'.format(function_name))
    
    plt.legend()
    plt.savefig('{}-results.png'.format(plot_file_name))

def save_to_csv(csv_file_name: str, row: list[float]): 
    csv_file = open('{}.csv'.format(csv_file_name), 'a')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)

    csv_file.close()

def plot_2d_map(values: np.ndarray, zlabel: str, extent: list[float], plot_file_name: str) -> None:
    plt.figure()
    im = plt.imshow(values, cmap=plt.cm.RdBu, extent=extent)
    plt.colorbar(im, label=zlabel)
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.savefig('{}-results.png'.format(plot_file_name))
    plt.close()

def save_results_variable_n(function_name: str, csv_row: list[float], csv_file_name: str,
                            plot_file_name: str, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    save_prediction_plot_from_points(function_name, plot_file_name,
                                                 x, x, y_true, y_true, y_pred,
                                                 train_label='Testing points')
        
    test_metric = dde.metrics.l2_relative_error(y_true, y_pred)
    csv_row += [test_metric]
    save_to_csv(csv_file_name, csv_row)

def save_results_variable_n_function(n: int, net: FunctionContinuousN, psi: callable, x_min: float, x_max: float, num_points: int,
                                layers: int, nodes: int, num_train: int, num_test: int, csv_file_name: str) -> None:
    x, y_pred, y_true = get_values_for_n(n, net, psi, x_min, x_max, num_points)
    function_name = '$\psi_{}$(x)'.format(n)
    plot_file_name = '{}-{}-{}-{}'.format(n, layers, nodes, num_train)    
    csv_row = [n, layers, nodes, num_train, num_test]

    save_results_variable_n(function_name, csv_row, csv_file_name, plot_file_name, x, y_true, y_pred)

def save_results_variable_n_pde(n: int, net: PDEApproach, psi: callable, x_min: float, x_max: float, num_points: int,
                                layers: int, nodes: int, num_train: int, num_test: int, is_random: bool, weights: int,
                                csv_file_name: str) -> None:
    x, y_pred, y_true = get_values_for_n(n, net, psi, x_min, x_max, num_points)
    function_name = '$\psi_{}$(x)'.format(n)
    plot_file_name = '{}-{}-{}-{}-{}-{}'.format(n, layers, nodes, num_train, is_random, weights)
    csv_row = [n, layers, nodes, num_train, num_test, is_random, weights]

    save_results_variable_n(function_name, csv_row, csv_file_name, plot_file_name, x, y_true, y_pred)

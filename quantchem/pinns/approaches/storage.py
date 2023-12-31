import csv
import numpy as np
import matplotlib.pyplot as plt


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

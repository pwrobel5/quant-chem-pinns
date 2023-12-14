import typing
import numpy as np

import quantchem.pinns.approaches.function.abstractfunction as function

# TODO: correct type annotations


def value_2d_rectangle(data: np.ndarray, psi_x: typing.Callable, psi_y: typing.Callable) -> np.ndarray:
    x_values = data[:, 0:1]
    y_values = data[:, 1:2]

    x_component = psi_x(x_values)
    y_component = psi_y(y_values)

    return x_component * y_component

def value_2d_model(data: np.ndarray, x_model, y_model) -> np.ndarray:
    x_values = data[:, 0:1]
    y_values = data[:, 1:2]

    x_component = x_model.predict(x_values)
    y_component = y_model.predict(y_values)

    return x_component * y_component

def get_2d_values(input_function, x_min, x_max, y_min, y_max, grid_dist=0.02) -> np.ndarray:
    x_ax = np.arange(x_min, x_max, grid_dist)
    y_ax = np.arange(y_min, y_max, grid_dist)
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

def get_values_for_n(n, model: function.FunctionApproach, function, x_min, x_max, num_points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, num=num_points)
    points = np.array(list([i, n] for i in x))  # list used to evaluate the generator, otherwise predict methods throws errors
    y_pred = model.predict(points)
    y_true = function(points)

    return x, y_pred, y_true
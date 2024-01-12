import numpy as np
import deepxde as dde

import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.fixedn as fixedn
import quantchem.pinns.problems.linearwell as linearwell
import quantchem.pinns.utils as utils


LX = 2
LY = 2


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem_x = linearwell.LinearWell(training_parameters.n, LX, 0)
    problem_y = linearwell.LinearWell(training_parameters.l, LY, 0)

    function_net_x = fixedn.FunctionFixedN(problem_x.exact_solution,
                                           problem_x.domain,
                                           training_parameters)
    function_net_y = fixedn.FunctionFixedN(problem_y.exact_solution,
                                           problem_y.domain,
                                           training_parameters)
    
    function_net_x.train_net()
    function_net_y.train_net()   

    true_values = utils.get_2d_values(
        lambda data: utils.value_2d_rectangle(data, problem_x.exact_solution, problem_y.exact_solution),
        -LX / 2,
        LX / 2,
        -LY / 2,
        LY / 2
    )
    
    predicted_values = utils.get_2d_values(
        lambda x: utils.value_2d_model(x, function_net_x, function_net_y),
        -LX / 2,
        LX / 2,
        -LY / 2,
        LY / 2
    )

    extent=[-LX / 2, LX / 2, -LY / 2, LY / 2]
    output_name_body = '{}-{}-{}-{}-{}'.format(training_parameters.n, 
                                               training_parameters.l, 
                                               training_parameters.layers, 
                                               training_parameters.nodes, 
                                               training_parameters.num_train)

    storage.plot_2d_map(
        predicted_values, 
        '$\psi_{{{}, {}}}$(x, y)'.format(training_parameters.n, training_parameters.l), 
        extent, 
        'prediction-{}'.format(output_name_body)
    )
    storage.plot_2d_map(
        np.abs(true_values - predicted_values), 
        '|$\psi_{{{},{}, predicted}}$(x, y) - $\psi_{{{},{}, true}}$(x, y)|'.format(training_parameters.n, training_parameters.l, 
                                                                                    training_parameters.n, training_parameters.l), 
        extent, 
        'difference-{}'.format(output_name_body)
    )

    test_metric = dde.metrics.l2_relative_error(true_values, predicted_values)
    csv_row = [training_parameters.n, training_parameters.l, 
               training_parameters.layers, training_parameters.nodes, 
               training_parameters.num_train, training_parameters.num_test, test_metric]
    storage.save_to_csv('rectangular-function', csv_row)

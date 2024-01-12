import numpy as np
import deepxde as dde

import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet
import quantchem.pinns.problems.linearwell as linearwell
import quantchem.pinns.utils as utils


LX = 2
LY = 2


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem_x = linearwell.LinearWell(training_parameters.n,
                                      LX,
                                      training_parameters.weight,
                                      training_parameters.is_random)
    problem_y = linearwell.LinearWell(training_parameters.l,
                                      LY,
                                      training_parameters.weight,
                                      training_parameters.is_random)
    
    pde_net_x = pdenet.PDEApproach(problem_x.exact_solution,
                                   problem_x.domain,
                                   problem_x.pde,
                                   problem_x.boundary_conditions,
                                   problem_x.loss_weights,
                                   training_parameters)
    pde_net_y = pdenet.PDEApproach(problem_y.exact_solution,
                                   problem_y.domain,
                                   problem_y.pde,
                                   problem_y.boundary_conditions,
                                   problem_y.loss_weights,
                                   training_parameters)
    
    pde_net_x.train_net()
    pde_net_y.train_net()

    true_values = utils.get_2d_values(
        lambda data: utils.value_2d_rectangle(data, problem_x.exact_solution, problem_y.exact_solution),
        -LX / 2,
        LX / 2,
        -LY / 2,
        LY / 2)

    predicted_values = utils.get_2d_values(
        lambda x: utils.value_2d_model(x, pde_net_x, pde_net_y),
        -LX / 2,
        LX / 2,
        -LY / 2,
        LY / 2
    )

    extent = [-LX / 2, LX / 2, -LY / 2, LY / 2]
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
    storage.save_to_csv('rectangular-pde', csv_row)

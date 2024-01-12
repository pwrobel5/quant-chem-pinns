import deepxde as dde
import numpy as np

import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet
import quantchem.pinns.problems.linearwell as linearwell
import quantchem.pinns.utils as utils


NX_MAX = 5
NY_MAX = 5

LX = 2
LY = 2


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem_x = linearwell.LinearWellVariableN(1, NX_MAX, LX, training_parameters.weight, training_parameters.is_random, use_quantum_1d=True)
    problem_y = linearwell.LinearWellVariableN(1, NY_MAX, LY, training_parameters.weight, training_parameters.is_random, use_quantum_1d=True)

    pde_net_x = pdenet.PDEApproach(problem_x.exact_solution,
                                   problem_x.domain,
                                   problem_x.pde,
                                   problem_x.boundary_conditions,
                                   problem_x.loss_weights,
                                   training_parameters,
                                   input_dimension=2)
    pde_net_y = pdenet.PDEApproach(problem_y.exact_solution,
                                   problem_y.domain,
                                   problem_y.pde,
                                   problem_y.boundary_conditions,
                                   problem_y.loss_weights,
                                   training_parameters,
                                   input_dimension=2)
    
    pde_net_x.train_net()
    pde_net_y.train_net()

    extent = [-LX / 2, LX / 2, -LY / 2, LY / 2]

    for nx in range(1, 8):
        for ny in range(1, 8):
            output_name_body = '{}-{}-{}-{}-{}'.format(nx, ny, 
                                               training_parameters.layers, 
                                               training_parameters.nodes, 
                                               training_parameters.num_train)
            _, predicted_x, true_x = utils.get_values_for_n(nx, pde_net_x, problem_x.exact_solution, -LX / 2, LX / 2, training_parameters.num_test)
            _, predicted_y, true_y = utils.get_values_for_n(ny, pde_net_y, problem_y.exact_solution, -LY / 2, LY / 2, training_parameters.num_test)
            
            true_values = true_x @ true_y.T
            predicted_values = predicted_x @ predicted_y.T
          
            storage.plot_2d_map(
                predicted_values,
                '$\psi_{{{}, {}}}$(x, y)'.format(nx, ny),
                extent,
                'prediction-{}'.format(output_name_body))
            storage.plot_2d_map(
                np.abs(true_values - predicted_values), 
                '|$\psi_{{{},{}, predicted}}$(x, y) - $\psi_{{{},{}, true}}$(x, y)|'.format(nx, ny, nx, ny), 
                extent, 
                'difference-{}'.format(output_name_body)
            )
            
            test_metric = dde.metrics.l2_relative_error(true_values, predicted_values)
            csv_row = [nx, ny, 
               training_parameters.layers, training_parameters.nodes, 
               training_parameters.num_train, training_parameters.num_test, test_metric]
            storage.save_to_csv('rectangular-quantum-1d-n', csv_row)

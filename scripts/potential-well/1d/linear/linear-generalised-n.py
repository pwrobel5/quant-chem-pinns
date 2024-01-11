import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet
import quantchem.pinns.problems.linearwell as linearwell


N_MAX = 5

L = 2


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem = linearwell.LinearWellVariableN(1, N_MAX, L, training_parameters.weight, training_parameters.is_random)

    pde_net = pdenet.PDEApproach(problem.exact_solution, 
                                 problem.domain, 
                                 problem.pde, 
                                 problem.boundary_conditions, 
                                 problem.loss_weights,
                                 training_parameters,
                                 input_dimension=2)
    pde_net.train_net()

    storage.save_loss_plot('linear-generalised-n')

    for i in range(1, 8):
        storage.save_results_variable_n_pde(i, 
                                            pde_net, 
                                            problem.exact_solution, 
                                            -L / 2, L / 2, 
                                            training_parameters.num_test,
                                            training_parameters.layers, 
                                            training_parameters.nodes, 
                                            training_parameters.num_train, 
                                            training_parameters.num_test, 
                                            training_parameters.is_random, 
                                            training_parameters.weight,
                                            'linear-generalised-n')

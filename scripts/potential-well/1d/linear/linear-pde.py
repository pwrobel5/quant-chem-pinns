import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet
import quantchem.pinns.problems.linearwell as linearwell


L = 2


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem = linearwell.LinearWell(training_parameters.n, 
                                    L, 
                                    training_parameters.weight, 
                                    training_parameters.is_random)

    pde_net = pdenet.PDEApproach(problem.exact_solution,
                                 problem.domain,
                                 problem.pde,
                                 problem.boundary_conditions,
                                 problem.loss_weights,
                                 training_parameters)
    pde_net.train_net()

    storage.save_loss_plot('linear-pde')

    function_name = '$\psi_{}$(x)'.format(training_parameters.n)
    plot_file_name = '{}-{}-{}-{}-{}-{}'.format(training_parameters.n, 
                                                training_parameters.layers, 
                                                training_parameters.nodes, 
                                                training_parameters.num_train, 
                                                training_parameters.is_random, 
                                                training_parameters.weight)
    storage.save_prediction_plot(function_name, plot_file_name)
    
    test_metric = pde_net.get_test_metric()
    csv_row = [training_parameters.n, training_parameters.layers, training_parameters.nodes, training_parameters.num_train, training_parameters.num_test, training_parameters.is_random, training_parameters.weight, test_metric]
    storage.save_to_csv('linear-pde', csv_row)

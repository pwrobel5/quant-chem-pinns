import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet
import quantchem.pinns.problems.circularwell as circularwell


R = 1


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem = circularwell.CircularWell(training_parameters.n,
                                        training_parameters.l,
                                        R,
                                        training_parameters.weight)
    
    pde_net = pdenet.PDEApproach(problem.exact_solution,
                                 problem.domain,
                                 problem.pde,
                                 problem.boundary_conditions,
                                 problem.loss_weights,
                                 training_parameters,
                                 exclusions=problem.exclusions)
    pde_net.train_net()

    storage.save_loss_plot('circular-pde')

    function_name = '$\psi_{{{},{}}}$(x)'.format(training_parameters.n, training_parameters.l)
    plot_file_name = '{}-{}-{}-{}-{}'.format(training_parameters.n, 
                                             training_parameters.layers, 
                                             training_parameters.nodes, 
                                             training_parameters.num_train, 
                                             training_parameters.weight)
    storage.save_prediction_plot(function_name, plot_file_name)

    test_metric = pde_net.get_test_metric()
    csv_row = [
        training_parameters.n, 
        training_parameters.layers, 
        training_parameters.nodes, 
        training_parameters.num_train, 
        training_parameters.num_test, 
        training_parameters.weight, 
        test_metric
    ]
    storage.save_to_csv('circular-pde', csv_row)

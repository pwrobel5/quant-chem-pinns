import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet
import quantchem.pinns.problems.harmonicoscillator as harmonicoscillator


R_max = 5
m = 1
omega = 0.5


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem = harmonicoscillator.HarmonicOscillator(training_parameters.n,
                                                    R_max,
                                                    m,
                                                    omega,
                                                    training_parameters.weight)
    
    pde_net = pdenet.PDEApproach(problem.exact_solution,
                                 problem.domain,
                                 problem.pde,
                                 problem.boundary_conditions,
                                 problem.loss_weights,
                                 training_parameters)
    pde_net.train_net()

    storage.save_loss_plot('pde')

    function_name = '$\psi_{}$(x)'.format(training_parameters.n)
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
    storage.save_to_csv('pde', csv_row)

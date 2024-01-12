import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.fixedn as fixedn
import quantchem.pinns.problems.circularwell as circularwell


R = 1


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem = circularwell.CircularWell(training_parameters.n, training_parameters.l, R, 0)

    function_net = fixedn.FunctionFixedN(problem.exact_solution,
                                         problem.domain,
                                         training_parameters)
    function_net.train_net()

    storage.save_loss_plot('circular-function')

    function_name = '$\psi_{{{},{}}}$(x)'.format(training_parameters.n, training_parameters.l)
    plot_file_name = '{}-{}-{}-{}-{}'.format(training_parameters.n, 
                                             training_parameters.l, 
                                             training_parameters.layers, 
                                             training_parameters.nodes, 
                                             training_parameters.num_train)
    storage.save_prediction_plot(function_name, plot_file_name)

    test_metric = function_net.get_test_metric()
    csv_row = [training_parameters.n, training_parameters.l, training_parameters.layers, training_parameters.nodes, training_parameters.num_train, training_parameters.num_test, test_metric]
    storage.save_to_csv('circular-function', csv_row)

import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.continuousn as continuousn
import quantchem.pinns.problems.linearwell as linearwell


N_MAX = 5

L = 2


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem = linearwell.LinearWellVariableN(1, N_MAX, L, 0)

    function_net = continuousn.FunctionContinuousN(problem.exact_solution, 
                                                   problem.domain, 
                                                   training_parameters)
    function_net.train_net()

    storage.save_loss_plot('linear-function-generalised-n')
    
    for i in range(1, 8):
        storage.save_results_variable_n_function(i, 
                                                 function_net, 
                                                 problem.exact_solution, 
                                                 -L / 2, L / 2, 
                                                 training_parameters.num_test, 
                                                 training_parameters.layers, 
                                                 training_parameters.nodes, 
                                                 training_parameters.num_train, 
                                                 training_parameters.num_test, 
                                                 'linear-function-generalised-n')

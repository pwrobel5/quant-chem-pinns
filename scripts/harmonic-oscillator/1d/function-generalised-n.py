import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.function.continuousn as continuousn
import quantchem.pinns.problems.harmonicoscillator as harmonicoscillator


R_max = 5
m = 1
omega = 0.5
N_MAX = 6


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem = harmonicoscillator.HarmonicOscillatorVariableN(0, N_MAX, R_max, m, omega, 0)

    function_net = continuousn.FunctionContinuousN(problem.exact_solution,
                                                   problem.domain,
                                                   training_parameters)
    function_net.train_net()
    
    storage.save_loss_plot('function-generalised-n')
    
    for i in range(0, N_MAX + 1):
        storage.save_results_variable_n_function(i,
                                                 function_net,
                                                 problem.exact_solution,
                                                 -R_max, R_max,
                                                 training_parameters.num_test,
                                                 training_parameters.layers,
                                                 training_parameters.nodes,
                                                 training_parameters.num_train,
                                                 training_parameters.num_test,
                                                 'function-generalised-n')

import quantchem.pinns.approaches.parameters as parameters
import quantchem.pinns.approaches.storage as storage
import quantchem.pinns.approaches.pde.pde as pdenet
import quantchem.pinns.problems.harmonicoscillator as harmonicoscillator


R_max = 5
m = 1
omega = 0.5
N_MAX = 6


if __name__ == '__main__':
    training_parameters = parameters.TrainingParameters()
    training_parameters.parse_arguments()

    problem = harmonicoscillator.HarmonicOscillatorVariableN(0, N_MAX, R_max, m, omega, training_parameters.weight)

    pde_net = pdenet.PDEApproach(problem.exact_solution,
                                 problem.domain,
                                 problem.pde,
                                 problem.boundary_conditions,
                                 problem.loss_weights,
                                 training_parameters,
                                 input_dimension=2)
    pde_net.train_net()

    storage.save_loss_plot('generalised-n')

    for i in range(0, N_MAX + 1):
        storage.save_results_variable_n_pde(i,
                                            pde_net,
                                            problem.exact_solution,
                                            -R_max, R_max,
                                            training_parameters.num_test,
                                            training_parameters.layers,
                                            training_parameters.nodes,
                                            training_parameters.num_train,
                                            training_parameters.num_test,
                                            training_parameters.is_random,
                                            training_parameters.weight,
                                            'generalised-n')

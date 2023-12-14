import deepxde as dde
import numpy as np
import typing
from abc import ABC

import quantchem.pinns.approaches.defaults as defaults


class FunctionApproach(ABC):
    
    def __init__(self,
                 function: typing.Callable[[np.ndarray], np.ndarray],
                 domain: dde.geometry.Geometry, 
                 layers: int = defaults.DEFAULT_LAYERS, 
                 nodes: int = defaults.DEFAULT_NODES, 
                 num_train: int = defaults.DEFAULT_NUM_TRAIN, 
                 num_test: int = defaults.DEFAULT_NUM_TEST,
                 input_dimension: int = 1) -> None:
        
        dde.config.set_random_seed(1234)
        
        self.data = dde.data.Function(
            domain,
            function,
            num_train,
            num_test
        )

        self.net = dde.nn.FNN(
            [input_dimension] + [nodes] * layers + [1],
            defaults.ACTIVATION,
            defaults.INITIALIZER
        )

        self.model = dde.Model(self.data, self.net)
        self.model.compile(defaults.OPTIMIZER, metrics=defaults.METRICS)
        self.test_metric = None

    def train_net(self) -> None:
        self.loss_history, self.train_state = self.model.train(iterations=defaults.ITERATIONS)
        self.test_metric = self.loss_history.metrics_test[-1][0]
        dde.saveplot(self.loss_history, self.train_state, issave=True, isplot=False)

    def get_test_metric(self) -> float:
        if self.test_metric is None:
            raise Exception('No metric available, the PINN is not trained!')
    
        return self.test_metric

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.model.predict(input)

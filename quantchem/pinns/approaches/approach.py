import abc
import deepxde as dde
import numpy as np

import quantchem.pinns.approaches.defaults as defaults


class AbstractApproach(abc.ABC):

    def __init__(self) -> None:
        
        dde.config.set_random_seed(1234)

        self._data = None
        self._net = None
        self._model = None
        self._test_metric = None
    
    def _compile_model(self,
                        layers: int,
                        nodes: int,                        
                        input_dimension: int,
                        loss_weights: list[float] = None) -> None:
        if self._data is None:
            raise Exception('Data field is not initialized!')

        self._net = dde.nn.FNN(
            [input_dimension] + [nodes] * layers + [1],
            defaults.ACTIVATION,
            defaults.INITIALIZER
        )

        self._model = dde.Model(self._data, self._net)
        self._model.compile(
            defaults.OPTIMIZER,
            metrics=defaults.METRICS,
            loss_weights=loss_weights
        )
    
    def train_net(self) -> None:
        self._loss_history, self._train_state = self._model.train(iterations=defaults.ITERATIONS)
        self._test_metric = self._loss_history.metrics_test[-1][0]
        dde.saveplot(self._loss_history, self._train_state, issave=True, isplot=False)

    def get_test_metric(self) -> float:
        if self._test_metric is None:
            raise Exception('No metric available, the PINN is not trained!')
    
        return self._test_metric

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self._model.predict(input)

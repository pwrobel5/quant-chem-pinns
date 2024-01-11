import deepxde as dde
import numpy as np
import typing

from ..approach import AbstractApproach
from ..parameters import TrainingParameters


class PDEApproach(AbstractApproach):

    def __init__(self,
                 solution: typing.Callable[[np.ndarray], np.ndarray],
                 domain: dde.geometry.Geometry,
                 pde: typing.Callable[[typing.Any], typing.Any],
                 boundary_conditions: list[dde.icbc.BC | dde.icbc.PointSetBC],
                 loss_weights: list[float],
                 parameters: TrainingParameters,
                 input_dimension: int = 1,
                 exclusions: list[float] = None) -> None:
        
        super().__init__()

        self._data = dde.data.PDE(
            domain,
            pde,
            boundary_conditions,
            num_domain=parameters.num_train,
            num_boundary=parameters.num_boundary,
            num_test=parameters.num_test,
            solution=solution,
            exclusions=exclusions
        )

        self._compile_model(parameters.layers, parameters.nodes, input_dimension, loss_weights)

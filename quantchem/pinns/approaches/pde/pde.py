import deepxde as dde
import numpy as np
import typing

from ..approach import AbstractApproach


class PDEApproach(AbstractApproach):

    def __init__(self,
                 solution: typing.Callable[[np.ndarray], np.ndarray],
                 domain: dde.geometry.Geometry,
                 pde: typing.Callable[[typing.Any], typing.Any],
                 boundary_conditions: list[dde.icbc.BC | dde.icbc.PointSetBC],
                 loss_weights: list[float],
                 layers: int,
                 nodes: int,
                 num_train: int,
                 num_boundary: int,
                 num_test: int,
                 input_dimension: int = 1) -> None:
        
        super().__init__()

        self._data = dde.data.PDE(
            domain,
            pde,
            boundary_conditions,
            num_domain=num_train,
            num_boundary=num_boundary,
            num_test=num_test,
            solution=solution
        )

        self._compile_model(layers, nodes, input_dimension, loss_weights)

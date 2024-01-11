import deepxde as dde
import numpy as np
import typing

from ..approach import AbstractApproach
from ..parameters import TrainingParameters


class FunctionApproach(AbstractApproach):
    
    def __init__(self,
                 function: typing.Callable[[np.ndarray], np.ndarray],
                 domain: dde.geometry.Geometry, 
                 parameters: TrainingParameters,
                 input_dimension: int = 1) -> None:
        
        super().__init__()
        
        self._data = dde.data.Function(
            domain,
            function,
            parameters.num_train,
            parameters.num_test
        )

        self._compile_model(parameters.layers, parameters.nodes, input_dimension)

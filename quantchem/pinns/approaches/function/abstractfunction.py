import deepxde as dde
import numpy as np
import typing

import quantchem.pinns.approaches.defaults as defaults
from ..approach import AbstractApproach


class FunctionApproach(AbstractApproach):
    
    def __init__(self,
                 function: typing.Callable[[np.ndarray], np.ndarray],
                 domain: dde.geometry.Geometry, 
                 layers: int = defaults.DEFAULT_LAYERS, 
                 nodes: int = defaults.DEFAULT_NODES, 
                 num_train: int = defaults.DEFAULT_NUM_TRAIN, 
                 num_test: int = defaults.DEFAULT_NUM_TEST,
                 input_dimension: int = 1) -> None:
        
        super().__init__()
        
        self._data = dde.data.Function(
            domain,
            function,
            num_train,
            num_test
        )

        self._compile_model(layers, nodes, input_dimension)

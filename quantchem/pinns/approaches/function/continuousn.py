import deepxde as dde
import numpy as np
import typing

import quantchem.pinns.approaches.defaults as defaults
from .abstractfunction import FunctionApproach


class FunctionContinuousN(FunctionApproach):

    def __init__(self, 
                 function: typing.Callable[[np.ndarray], np.ndarray],
                 domain: dde.geometry.Geometry, 
                 layers: int = defaults.DEFAULT_LAYERS, 
                 nodes: int = defaults.DEFAULT_NODES, 
                 num_train: int = defaults.DEFAULT_NUM_TRAIN, 
                 num_test: int = defaults.DEFAULT_NUM_TEST) -> None:
        
        super().__init__(function,
                         domain,
                         layers,
                         nodes,
                         num_train,
                         num_test,
                         input_dimension=2)

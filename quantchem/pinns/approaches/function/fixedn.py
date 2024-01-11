import deepxde as dde
import numpy as np
import typing

from .abstractfunction import FunctionApproach
from ..parameters import TrainingParameters


class FunctionFixedN(FunctionApproach):

    def __init__(self, 
                 function: typing.Callable[[np.ndarray], np.ndarray],
                 domain: dde.geometry.Geometry, 
                 parameters: TrainingParameters) -> None:
        
        super().__init__(function,
                         domain,
                         parameters.layers,
                         parameters.nodes,
                         parameters.num_train,
                         parameters.num_test,
                         input_dimension=1)

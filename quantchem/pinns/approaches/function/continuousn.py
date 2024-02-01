import deepxde as dde
import numpy as np
import typing

from .abstractfunction import FunctionApproach
from ..parameters import TrainingParameters


class FunctionContinuousN(FunctionApproach):

    def __init__(self, 
                 function: typing.Callable[[np.ndarray], np.ndarray],
                 domain: dde.geometry.Geometry,
                 parameters: TrainingParameters) -> None:
        
        super().__init__(function,
                         domain,
                         parameters,
                         input_dimension=2)

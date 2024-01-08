import abc
import deepxde as dde
import numpy as np


class AbstractProblem(abc.ABC):
    
    def __init__(self) -> None:        
        pass

    @abc.abstractmethod
    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented('exact_solution method not implemented!')

    @abc.abstractmethod
    def pde(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplemented('pde method not implemented!')
    
    @abc.abstractproperty
    def domain(self) -> dde.geometry:
        raise NotImplemented('domain property not implemented!')

    @abc.abstractproperty
    def boundary_conditions(self) -> list[dde.icbc.BC | dde.icbc.PointSetBC]:
        raise NotImplemented('boundary_conditions property not implemented!')

    @abc.abstractproperty
    def loss_weights(self) -> list[float]:
        raise NotImplemented('loss_weights not implemented!')

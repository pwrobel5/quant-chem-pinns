import numpy as np
import deepxde as dde
from numpy import ndarray

from .abstractproblem import AbstractProblem
from ..approaches.geometry.quantum1d import Quantum1D


class LinearWell(AbstractProblem):

    def __init__(self, n: int, L: float, loss_weight: float, random_collocation: bool = True) -> None:
        self._n = n
        self._L = L

        self._domain = dde.geometry.Interval(-self._L / 2, self._L / 2)
        
        collocation_points = self.__get_random_collocation_points() if random_collocation else self.__get_extremal_collocation_points()
        collocation_values = self.exact_solution(collocation_points)

        point_set_bc = dde.icbc.PointSetBC(collocation_points, collocation_values)
        dirichlet_bc = dde.icbc.DirichletBC(self._domain, lambda x: 0, self.__x_boundary)
        
        self._boundary_conditions = [point_set_bc, dirichlet_bc]
        self._loss_weights = [1, loss_weight, loss_weight]
    
    def exact_solution(self, x: np.ndarray) -> ndarray:
        k = (self._n * np.pi) / self._L
        normalization_constant = np.sqrt(2.0 / self._L)
        return normalization_constant * np.sin(k * (x + 0.5 * self._L))
    
    def pde(self, x: ndarray, y: ndarray) -> ndarray:
        k = (self._n * np.pi) / self._L
        E = 0.5 * (k ** 2)
        dy_xx = dde.grad.hessian(y, x)
        return 0.5 * dy_xx + E * y
    
    @property
    def domain(self) -> dde.geometry:
        return self._domain
    
    @property
    def boundary_conditions(self) -> list[dde.icbc.BC | dde.icbc.PointSetBC]:
        return self._boundary_conditions
    
    @property
    def loss_weights(self) -> list[float]:
        return self._loss_weights

    def __get_extremal_collocation_points(self) -> np.ndarray:
        points = []
        for k in range(self._n):
            x = (k * self._L) / self._n + self._L / (2 * self._n) - self._L / 2
            points.append([x])
            
        if self._n == 1:
            points.append([self._L / 4])
            points.append([-self._L / 4])
        
        return np.array(points)

    def __get_random_collocation_points(self) -> np.ndarray:
        points = []
        for _ in range(self._n):
            x = np.random.uniform(low=-self._L/2, high=self._L/2)
            points.append([x])
        
        if self._n == 1:
            points.append([np.random.uniform(low=-self._L/2, high=self._L/2)])
            points.append([np.random.uniform(low=-self._L/2, high=self._L/2)])
        
        return np.array(points)
    
    def __x_boundary(self, _, on_boundary: bool) -> bool:
        return on_boundary
    

class LinearWellVariableN(AbstractProblem):

    def __init__(self, n_min: int, n_max: int, L: float, loss_weight: float, random_collocation: bool = True, use_quantum_1d: bool = False) -> None:
        self._n_min = n_min
        self._n_max = n_max
        self._L = L

        if use_quantum_1d:
            self._domain = Quantum1D(-self._L / 2, self._L / 2, self._n_min, self._n_max)
        else:
            self._domain = dde.geometry.Rectangle([-self._L / 2, self._n_min], [self._L / 2, self._n_max])
        
        collocation_points = self.__get_random_collocation_points() if random_collocation else self.__get_extremal_collocation_points()
        collocation_values = self.exact_solution(collocation_points)

        point_set_bc = dde.icbc.PointSetBC(collocation_points, collocation_values)
        dirichlet_bc = dde.icbc.DirichletBC(self._domain, lambda x: 0, self.__x_boundary)
        
        self._boundary_conditions = [point_set_bc, dirichlet_bc]
        self._loss_weights = [1, loss_weight, loss_weight]
    
    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        normalization_constant = np.sqrt(2.0 / self._L)
        n = x[:, 1:2]
        k = (n * np.pi) / self._L
        return normalization_constant * np.sin(k * (x[:, 0:1] + (self._L / 2)))
    
    def pde(self, x: ndarray, y: ndarray) -> ndarray:
        n = x[:, 1:2]
        k = (n * np.pi) / self._L
        E = 0.5 * (k ** 2)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return 0.5 * dy_xx + E * y

    @property
    def domain(self) -> dde.geometry:
        return self._domain
    
    @property
    def boundary_conditions(self) -> list[dde.icbc.BC | dde.icbc.PointSetBC]:
        return self._boundary_conditions
    
    @property
    def loss_weights(self) -> list[float]:
        return self._loss_weights
    
    def __get_extremal_collocation_points(self) -> np.ndarray:
        points = []
        for n in range(self._n_min, self._n_max + 1):
            for k in range(n):
                x = (k * self._L) / n + self._L / (2 * n) - self._L / 2
                points.append((x, n))
                
            if n == 1:
                points.append((self._L / 4, n))
                points.append((-self._L / 4, n))
        
        return np.array(points)

    def __get_random_collocation_points(self) -> np.ndarray:
        points = []
        for n in range(self._n_min, self._n_max + 1):
            for _ in range(n):
                x = np.random.uniform(low=-self._L/2, high=self._L/2)
                points.append((x, n))
            
            if n == 1:
                points.append((np.random.uniform(low=-self._L/2, high=self._L/2), n))
                points.append((np.random.uniform(low=-self._L/2, high=self._L/2), n))
        
        return np.array(points)

    def __x_boundary(self, x, _) -> bool:
        if x[1].is_integer():
            return np.isclose(-self._L / 2, x[0]) or np.isclose(self._L / 2, x[0])
        return False

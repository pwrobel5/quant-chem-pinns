import deepxde as dde
import numpy as np
import scipy.special as sp

from .abstractproblem import AbstractProblem


class TriangularWell(AbstractProblem):

    def __init__(self, n: int, L: float, q: float, epsilon: float, loss_weight: float) -> None:
        self._n = n
        self._L = L
        self._q = q
        self._epsilon = epsilon

        self._domain = dde.geometry.Interval(0, self._L)
        
        collocation_points = self.__get_collocation_points()
        collocation_values = self.exact_solution(collocation_points)

        point_set_bc = dde.icbc.PointSetBC(collocation_points, collocation_values)
        dirichlet_bc = dde.icbc.DirichletBC(self._domain, lambda x: 0, self.__x_boundary)
        
        self._boundary_conditions = [point_set_bc, dirichlet_bc]
        self._loss_weights = [1, loss_weight, loss_weight]
    
    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        ai_zeros, _, _, _ = sp.ai_zeros(self._n)
        alpha_n = ai_zeros[-1]
        E = -alpha_n * np.power((self._q ** 2) * (self._epsilon ** 2) / 2, (1 / 3))
        
        cube_root = np.power(2 * self._q * self._epsilon, (1 / 3))
        
        ai_value, _, _, _ = sp.airy(cube_root * (x - (E / (self._q * self._epsilon))))
        _, aip_value, _, _ = sp.airy(-cube_root * (E / (self._q * self._epsilon)))
        
        normalization_constant = np.sqrt(cube_root / (aip_value ** 2))
        
        return normalization_constant * ai_value
    
    def pde(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ai_zeros, _, _, _ = sp.ai_zeros(self._n)
        alpha_n = ai_zeros[-1]
        E = -alpha_n * np.power((self._q ** 2) * (self._epsilon ** 2) / 2, (1 / 3))
        
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        
        return dy_xx + 2 * (E - self._q * self._epsilon * x) * y
    
    @property
    def domain(self) -> dde.geometry:
        return self._domain

    @property
    def boundary_conditions(self) -> list[dde.icbc.BC | dde.icbc.PointSetBC]:
        return self._boundary_conditions

    @property
    def loss_weights(self) -> list[float]:
        return self._loss_weights
    
    @property
    def exclusions(self) -> list[float]:
        return [self._L]

    def __get_collocation_points(self) -> np.ndarray:
        rs = []
        
        r_interval = self._L / (3 * self._n)
        
        for k in range(3 * self._n - 1):
            r = (k + 1) * r_interval
            rs.append(r)
        
        return np.array(rs).reshape((3 * self._n - 1, 1))

    def __x_boundary(self, x: np.ndarray, on_boundary: bool) -> bool:
        return on_boundary and np.isclose(0, x[0])

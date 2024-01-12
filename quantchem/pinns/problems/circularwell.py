import deepxde as dde
import numpy as np
import scipy.special as sp

from .abstractproblem import AbstractProblem


class CircularWell(AbstractProblem):

    def __init__(self, n: int, l: int, R: float, loss_weight: float) -> None:
        self._n = n
        self._l = l
        self._R = R
        
        self._domain = dde.geometry.Interval(0, self._R)

        collocation_points = self.__get_collocation_points()
        collocation_values = self.exact_solution(collocation_points)
        
        point_set_bc = dde.icbc.PointSetBC(collocation_points, collocation_values)
        dirichlet_bc = dde.icbc.DirichletBC(self._domain, lambda x: 0, self.__x_boundary)
        neumann_bc = dde.icbc.NeumannBC(self._domain, self.__boundary_derivative_value, self.__x_boundary)

        self._boundary_conditions = [point_set_bc, dirichlet_bc, neumann_bc]
        self._loss_weights = [1, loss_weight, loss_weight, loss_weight]

    
    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        alpha = sp.jn_zeros(self._l, self._n)[-1]
        normalization_constant = np.sqrt(2) / (self._R * np.abs(sp.jv(self._l + 1, alpha)))
        bessel = sp.jv(self._l, alpha * (x / self._R))

        return normalization_constant * bessel
    
    def pde(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dy_r = dde.grad.jacobian(y, x, i=0, j=0)
        dy_rr = dde.grad.hessian(y, x, i=0, j=0)
        
        k = sp.jn_zeros(self._l, self._n)[-1] 
        
        return (x ** 2) * dy_rr + x * dy_r + ((x ** 2) * (k ** 2) - (self._l ** 2)) * y
    
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
        return [0.0]
    
    def __get_collocation_points(self) -> np.ndarray:
        rs = []
    
        r_interval = self._R / (3 * (self._n + self._l))
        
        for k in range(3 * (self._n + self._l) - 1):
            r = (k + 1) * r_interval
            rs.append(r)
        
        return np.array(rs).reshape((3 * (self._n + self._l) - 1, 1))
    
    def __boundary_derivative_value(self, x: np.ndarray) -> np.ndarray:
        alpha = sp.jn_zeros(self._l, self._n)[-1]
        normalization_constant = (np.sqrt(2) * alpha) / ((self._R ** 2) * np.abs(sp.jv(self._l + 1, alpha)))
        return normalization_constant * sp.jvp(self._l, alpha * x / self._R)

    def __x_boundary(self, _, on_boundary: bool) -> bool:
        return on_boundary

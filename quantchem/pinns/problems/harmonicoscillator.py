import deepxde as dde
import numpy as np
import scipy.special as sp

from .abstractproblem import AbstractProblem

class HarmonicOscillator(AbstractProblem):

    def __init__(self, n: int, R_max: float, m: float, omega: float, loss_weight: float) -> None:
        self._n = n
        self._R_max = R_max
        self._m = m
        self._omega = omega
        
        self._domain = dde.geometry.Interval(-self._R_max, self._R_max)

        if n % 2 == 0:
            dirichlet_bc = dde.icbc.DirichletBC(self._domain, self.__boundary_value, self.__x_middle)
        else:
            dirichlet_bc = dde.icbc.DirichletBC(self._domain, self.exact_solution, self.__x_boundary)
        neumann_bc = dde.icbc.NeumannBC(self._domain, self.__boundary_derivative_value, self.__x_middle)

        self._boundary_conditions = [dirichlet_bc, neumann_bc]
        self._loss_weights = [1, loss_weight, loss_weight]
    
    
    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        constants = (1.0 / (np.sqrt(np.math.factorial(self._n) * (2 ** self._n)))) * (((self._m * self._omega) / np.pi) ** 0.25)
        exponent = np.exp(-0.5 * self._m * self._omega * np.power(x, 2))
        hermite_coefficients = [0] * self._n + [1]
        hermite = np.polynomial.hermite.Hermite(hermite_coefficients)
        hermite_value = hermite(x * np.sqrt(self._m * self._omega))
        result = constants * exponent * hermite_value
    
        return result

    def pde(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dy_xx = dde.grad.hessian(y, x)
        E = (self._n + 0.5) * self._omega
        U = 0.5 * self._m * (self._omega ** 2) * (x ** 2)
        
        return -dy_xx / 2 + (U - E) * y
    
    @property
    def domain(self) -> dde.geometry:
        return self._domain
    
    @property
    def boundary_conditions(self) -> list[dde.icbc.BC | dde.icbc.PointSetBC]:
        return self._boundary_conditions
    
    @property
    def loss_weights(self) -> list[float]:
        return self._loss_weights
    
    def __boundary_value(self, x: np.ndarray) -> np.ndarray:
        if self._n % 2 == 1:
            return 0
        
        constants = (1.0 / (np.sqrt(np.math.factorial(self._n) * (2 ** self._n)))) * (((self._m * self._omega) / np.pi) ** 0.25)
        if self._n == 0:
            return constants
        
        hermite_coefficients = [0] * self._n + [1]
        hermite = np.polynomial.Hermite(hermite_coefficients)
        hermite_value = hermite(0)

        return constants * hermite_value

    def __boundary_derivative_value(self, x: np.array) -> np.array:
        if self._n % 2 == 0:
            return 0
            
        constants = 2 * self._n * (1.0 / (np.sqrt(np.math.factorial(self._n) * (2 ** self._n)))) * (1.0 / (np.pi ** 0.25)) * ((self._m * self._omega) ** 0.75)
        hermite_coefficients = [0] * (self._n - 1) + [1]
        hermite = np.polynomial.Hermite(hermite_coefficients)
        hermite_value = hermite(0)

        return constants * hermite_value

    def __x_boundary(self, x: float, _) -> bool:
        return np.isclose(x[0], self._R_max / 2) or np.isclose(x[0], -self._R_max / 2)

    def __x_middle(self, x: float, _) -> bool:
        return np.isclose(x[0], 0)

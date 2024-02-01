import deepxde as dde
import numpy as np
import scipy.special as sp

from .abstractproblem import AbstractProblem
from ..approaches.geometry.quantum1d import Quantum1D

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


class HarmonicOscillatorVariableN(AbstractProblem):
    
    def __init__(self, n_min: int, n_max: int, R_max: float, m: float, omega: float, loss_weight: float, use_quantum_1d: bool = False) -> None:
        self._n_min = n_min
        self._n_max = n_max
        self._R_max = R_max
        self._m = m
        self._omega = omega

        if use_quantum_1d:
            self._domain = Quantum1D(-self._R_max, self._R_max, self._n_min, self._n_max)
        else:
            self._domain = dde.geometry.Rectangle([-self._R_max, self._n_min], [self._R_max, self._n_max])
        
        collocation_points = self.__get_collocation_points()
        collocation_values = np.array([self.__boundary_value(x) for x in collocation_points])
        collocation_values = collocation_values.reshape(collocation_values.shape + (1,))

        derivative_values = np.array([self.__boundary_derivative_value(x) for x in collocation_points])
        derivative_values = derivative_values.reshape(derivative_values.shape + (1,))

        point_set_bc = dde.icbc.PointSetBC(collocation_points, collocation_values)
        operator_bc = dde.icbc.PointSetOperatorBC(collocation_points, derivative_values, self.__derivative_operator)

        self._boundary_conditions = [point_set_bc, operator_bc]
        self._loss_weights = [1, loss_weight, loss_weight]
    
    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        n = x[:, 1:2]
        factorials = []
        for n_val in n:
            factorials.append(sp.gamma(n_val + 1.0))
        factorials = np.array(factorials).reshape(len(factorials), 1)
        
        constants = (1.0 / (np.sqrt(factorials * (2 ** n)))) * (((self._m * self._omega) / np.pi) ** 0.25)

        exponent = np.exp(-0.5 * self._m * self._omega * np.power(x[:, 0:1], 2))
        
        hermite_values = []
        for index, n_val in enumerate(n):
            coefficients = int(n_val) * [0] + [1]
            hermite = np.polynomial.hermite.Hermite(coefficients)
            hermite_value = hermite(x[index, 0] * np.sqrt(self._m * self._omega))
            hermite_values.append(hermite_value)
        hermite_values = np.array(hermite_values).reshape(len(hermite_values), 1)

        result = constants * exponent * hermite_values
        return result
    
    def pde(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = x[:, 1:2]
        E = (n + 0.5) * self._omega
        U = 0.5 * self._m * (self._omega ** 2) * (x[:, 0:1] ** 2)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
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
    
    def __get_collocation_points(self) -> np.ndarray:
        points = []
        for n in range(0, self._n_max + 1):
            points.append((0.0, n))
        return np.array(points)

    def __boundary_value(self, x: np.ndarray) -> np.ndarray:
        n = x[-1]

        if n % 2 == 1:
            return 0
        
        constants = (1.0 / (np.sqrt(sp.gamma(n + 1) * (2 ** n)))) * (((self._m * self._omega) / np.pi) ** 0.25)
        if n == 0:
            return constants
        
        hermite_coefficients = [0] * int(n) + [1]
        hermite = np.polynomial.Hermite(hermite_coefficients)
        hermite_value = hermite(0)

        return constants * hermite_value

    def __boundary_derivative_value(self, x: np.ndarray) -> np.ndarray:
        n = x[-1]

        if n % 2 == 0:
            return 0
        
        constants = 2 * n * (1.0 / (np.sqrt(sp.gamma(n + 1) * (2 ** n)))) * (1.0 / (np.pi ** 0.25)) * ((self._m * self._omega) ** 0.75)
        hermite_coefficients = [0] * (int(n) - 1) + [1]
        hermite = np.polynomial.Hermite(hermite_coefficients)
        hermite_value = hermite(0)

        return constants * hermite_value

    def __derivative_operator(self, x: np.ndarray, y: np.ndarray, _) -> np.ndarray:
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        return dy_x

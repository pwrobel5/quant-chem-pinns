import numpy as np
import deepxde as dde
import deepxde.geometry.sampler as sampler


class Quantum1D(dde.geometry.geometry_1d.Interval):
    def __init__(self, x_min, x_max, n_min, n_max):
        super().__init__(x_min, x_max)
        self._n_min = n_min
        self._n_max = n_max
    
    def inside(self, x):
        x_inside = np.logical_and(self.l <= x[:, 0:1], x[:, 0:1] <= self.r).flatten()
        n_integer = np.array([n[0].is_integer for n in x[:, 1:2]]).reshape(x_inside.shape)
        return np.logical_and(x_inside, n_integer)
    
    def on_boundary(self, x):
        return super().on_boundary(x[:, 0:1])
    
    def distance2boundary(self, x, dirn):
        return super().distance2boundary(x[:, 0:1], dirn)
    
    def mindist2boundary(self, x):
        return super().mindist2boundary(x[:, 0:1])
    
    def boundary_normal(self, x):
        return super().boundary_normal(x[:, 0:1])
    
    def uniform_points(self, n, boundary=True):
        xs_per_n = n // self._n_max
        xs = super().uniform_points(xs_per_n, boundary=boundary)
        
        return self.__asign_ns_to_points(n, xs)
    
    def log_uniform_points(self, n, boundary=True):
        xs_per_n = n // self._n_max
        xs = super().log_uniform_points(xs_per_n, boundary=boundary)
        
        return self.__asign_ns_to_points(n, xs)        
    
    def random_points(self, n, random='pseudo'):
        random_points = []
        
        xs = sampler.sample(n, 1, random)
        xs = self.diam * xs + self.l
        for x in xs:
            i = np.random.randint(1, self._n_max + 1)
            random_points.append(np.hstack([x, i]))
              
        return np.array(random_points)
    
    def uniform_boundary_points(self, n):
        xs_per_n = n // self._n_max
        xs = super().uniform_boundary_points(xs_per_n)

        return self.__asign_ns_to_points(n, xs)
    
    def random_boundary_points(self, n, random='pseudo'):        
        xs_per_n = n // self._n_max
        random_boundary_points = []
        
        for i in range(1, self._n_max + 1):
            if xs_per_n == 2:
                random_boundary_points.append([self.l, i])
                random_boundary_points.append([self.r, i])
            else:
                xs = np.random.choice([self.l, self.r], xs_per_n)
                for x in xs:
                    random_boundary_points.append([x, i])
        
        rest = n - (xs_per_n * self._n_max)
        if rest != 0:
            print('WARNING: non zero rest points: {}'.format(rest))
        
        return np.array(random_boundary_points)
    
    def periodic_point(self, x, component=0):
        raise NotImplementedError('Periodic point not implemented')
    
    def background_points(self, x, dirn, dist2npt, shift):
        raise NotImplementedError('Background points not implemented')
    
    def __asign_ns_to_points(self, n, xs):
        xs_per_n = n // self._n_max
        uniform_points = []
        
        for i in range(1, self._n_max + 1):
            for x in xs:
                uniform_points.append(np.hstack([x, i]))
        rest = n - (xs_per_n * self._n_max)
        if rest != 0:
            print('WARNING: non zero rest points: {}'.format(rest))
               
        return np.array(uniform_points)

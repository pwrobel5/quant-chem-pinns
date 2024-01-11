import argparse
from .defaults import *

class TrainingParameters:
    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument('-n', '--quantum-number', default=DEFAULT_N)
        self._parser.add_argument('-l', '--secondary-quantum-number', default=DEFAULT_L)
        self._parser.add_argument('-ndense', '--layers', default=DEFAULT_LAYERS)
        self._parser.add_argument('-nnodes', '--nodes', default=DEFAULT_NODES)
        self._parser.add_argument('-ntrain', '--num-train', default=DEFAULT_NUM_TRAIN)
        self._parser.add_argument('-nboundary', '--num-boundary', default=DEFAULT_NUM_BOUNDARY)
        self._parser.add_argument('-ntest', '--num-test', default=DEFAULT_NUM_TEST)
        self._parser.add_argument('-weights', '--loss-weight', default=DEFAULT_LOSS_WEIGHTS)
        self._parser.add_argument('-random', '--random-collocation-points', action='store_true')
        self._args = None

    def parse_arguments(self) -> None:
        self._args = self._parser.parse_args()
    
    @property
    def n(self) -> int:
        return int(self._args.quantum_number)
    
    @property
    def l(self) -> int:
        return int(self._args.secondary_quantum_number)
    
    @property
    def layers(self) -> int:
        return int(self._args.layers)
    
    @property
    def nodes(self) -> int:
        return int(self._args.nodes)
    
    @property
    def num_train(self) -> int:
        return int(self._args.num_train)
    
    @property
    def num_boundary(self) -> int:
        return int(self._args.num_boundary)
    
    @property
    def num_test(self) -> int:
        return int(self._args.num_test)
    
    @property
    def weight(self) -> int:
        return int(self._args.loss_weight)
    
    @property
    def is_random(self) -> bool:
        return bool(self._args.random_collocation_points)
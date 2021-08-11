import numpy as np
from abc import ABC, abstractmethod

class OptimizationProblem(object):
    """Base class for optimization problems."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.dim = None
        self.lb = None
        self.ub = None

    def __check_input__(self, x):
        if len(x) != self.dim:
            raise ValueError("Dimension mismatch")

    @abstractmethod
    def eval(self):  # pragma: no cover
        pass


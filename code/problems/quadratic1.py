import numpy as np

from optimization_problem import OptimizationProblem


class Quadratic1(OptimizationProblem):
    """Quadratic ||(x-x^*)||^2
    """

    def __init__(self, dim=2):
        self.dim = dim
        self.min = 0
        self.minimum = np.ones(dim)
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

    def eval(self, x):
        self.__check_input__(x)
        return np.sum((x-self.minimum)**2)


    def grad(self,x):
        return 2*(x-self.minimum)


import numpy as np
from optimization_problem import OptimizationProblem

class Quadratic(OptimizationProblem):
    """Class for bound constrained quadratic optimization problems:
       min (x-x0) @ A @ (x-x0)
       s.t x in [lb,ub]

       If the bounds are infinite the problem must be convex.
       Otherwise the problem can be nonconvex, but must have 
       finite bounds.
       """

    def __init__(self,lb,ub,A,x0):
        self.dim = len(x0)
        self.lb = lb
        self.ub = ub
        self.A  = A
        self.x0 = x0
        self.minimum = None # generally no analytic solution

    def eval(self,x): 
        return (x-self.x0) @ self.A @ (x-self.x0) 

    def grad(self,x): 
        return (self.A + self.A.T) @ (x-self.x0)

    def hess(self,x):
        return (self.A + self.A.T)